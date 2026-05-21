import igraph as ig
from datetime import datetime
import hashlib, base64, json, os
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Iterable, Union, Any
from tqdm import tqdm
import numpy as np
import pandas as pd
import faiss
import json
import random

from src.config.base_config import logger
from src.config.rag_config import RAGConfig
from src.utils.llm_utils import embed_fn
from src.utils.common_utils import FileHelper


Record = Tuple[str, str, str, str, str]


def _clean_token(x: str) -> str:
    return x.replace("_", " ")


def _parse_line_4_or_5(line: str) -> Record:

    parts = [p.strip() for p in line.rstrip("\n").split("\t")]
    if len(parts) == 4:
        s, r, o, t = parts

        datetime.strptime(t, "%Y-%m-%d")
        return _clean_token(s), _clean_token(r), _clean_token(o), t, t
    elif len(parts) == 5:
        s, r, o, t1, t2 = parts

        return _clean_token(s), _clean_token(r), _clean_token(o), t1, t2
    else:
        raise ValueError(f"Bad line with {len(parts)} columns: {line!r}")


def load_records_from_txt(file_path: str) -> Iterable[Record]:
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            yield _parse_line_4_or_5(line)


def _b64u_sha1(s: str, prefix: str) -> str:
    h = hashlib.sha1(s.encode("utf-8")).digest()
    short = base64.urlsafe_b64encode(h).decode("ascii").rstrip("=")[:16]
    return f"{prefix}{short}"


def build_igraph_from_txt(file_path: str) -> Tuple[ig.Graph, Dict[str, Dict[str, int]]]:
    logger.info(f"Loading data from {file_path} ...")
    records = list(load_records_from_txt(file_path))  # (s, r, o, t1_str, t2_str)
    logger.info(f"Loaded {len(records):,} facts")


    entities, relations = set(), set()
    start_times, end_times = set(), set()
    timestamps = set() 

    def make_timestamp(t1: str, t2: str) -> str:

        return t1 if t1 == t2 else f"{t1}-{t2}"

    for s, r, o, t1, t2 in records:
        entities.update([s, o])
        relations.add(r)
        start_times.add(t1)
        end_times.add(t2)
        timestamps.add(make_timestamp(t1, t2))

    entity_list = sorted(entities)
    relation_list = sorted(relations)
    start_list = sorted(start_times)
    end_list = sorted(end_times)
    timestamp_list = sorted(timestamps)

    name2vid = {n: i for i, n in enumerate(entity_list)}
    rel2id = {r: i for i, r in enumerate(relation_list)}
    start2id = {t: i for i, t in enumerate(start_list)} 
    end2id = {t: i for i, t in enumerate(end_list)} 
    ts2id = {ts: i for i, ts in enumerate(timestamp_list)}

    node_ids = [_b64u_sha1(n, "ent_") for n in entity_list]

    g = ig.Graph(directed=True)
    g.add_vertices(len(entity_list))
    g.vs["name"] = entity_list
    g.vs["node_id"] = node_ids


    edge_tuples = []
    edge_relation, edge_relation_id = [], []

    edge_start_time, edge_end_time = [], []
    edge_start_id, edge_end_id = [], [] 
    edge_timestamp, edge_timestamp_id = [], [] 

    edge_src_node_id, edge_dst_node_id, edge_ids = [], [], []
    occ = defaultdict(int)

    print("Building edges ...")
    for s, r, o, t1, t2 in tqdm(records):
        u, v = name2vid[s], name2vid[o]
        edge_tuples.append((u, v))
        edge_relation.append(r)
        edge_relation_id.append(rel2id[r])


        ts = make_timestamp(t1, t2)
        edge_start_time.append(t1)
        edge_end_time.append(t2)
        edge_start_id.append(start2id[t1])
        edge_end_id.append(end2id[t2])
        edge_timestamp.append(ts)
        edge_timestamp_id.append(ts2id[ts])

        edge_src_node_id.append(node_ids[u])
        edge_dst_node_id.append(node_ids[v])

        idx = occ[(s, r, o, t1, t2)]
        occ[(s, r, o, t1, t2)] += 1
        eid = _b64u_sha1(f"{s}|{r}|{o}|{t1}|{t2}|{idx}", "evt_")
        edge_ids.append(eid)

    g.add_edges(edge_tuples)

    g.es["edge_id"] = edge_ids
    g.es["relation"] = edge_relation
    g.es["relation_id"] = edge_relation_id
    g.es["src_node_id"] = edge_src_node_id
    g.es["dst_node_id"] = edge_dst_node_id


    g.es["start_time"] = edge_start_time 
    g.es["end_time"] = edge_end_time 
    g.es["start_id"] = edge_start_id  
    g.es["end_id"] = edge_end_id
    g.es["timestamp"] = edge_timestamp  
    g.es["timestamp_id"] = edge_timestamp_id


    g["relation_vocab"] = relation_list
    g["start_time_vocab"] = start_list
    g["end_time_vocab"] = end_list
    g["timestamp_vocab"] = timestamp_list

    meta = {
        "name2vid": name2vid,
        "rel2id": rel2id,
        "start2id": start2id,
        "end2id": end2id,
        "ts2id": ts2id,
    }
    return g, meta



def _build_faiss_index(
    vecs: np.ndarray, use_cosine: bool, ids: Optional[np.ndarray] = None
) -> faiss.Index:

    assert vecs.dtype == np.float32 and vecs.ndim == 2
    n, d = vecs.shape
    index = faiss.IndexFlatIP(d) if use_cosine else faiss.IndexFlatL2(d)

    if ids is not None:
        assert ids.dtype == np.int64 and ids.shape == (n,)
        index = faiss.IndexIDMap2(index) 
        index.add_with_ids(vecs, ids)
    else:
        index.add(vecs)
    return index



def load_graph_and_indexes(base_store_dir: str):

    meta_path = os.path.join(base_store_dir, "meta.json")

    if not FileHelper.judge_file_exist(meta_path):
        raise FileNotFoundError(f"not found meta.json: {meta_path}")

    meta = FileHelper.load_json(meta_path)
    files = meta.get("files", {})

    graph_path = files.get("graph", os.path.join(base_store_dir, "graph.pkl"))
    entity_index_path = files.get(
        "entity_index", os.path.join(base_store_dir, "index/entity.index")
    )
    relation_index_path = files.get(
        "relation_index", os.path.join(base_store_dir, "index/relation.index")
    )
    timestamp_index_path = files.get(
        "timestamp_index", os.path.join(base_store_dir, "index/timestamp.index")
    )

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"not found graph file: {graph_path}")
    if not os.path.exists(entity_index_path):
        raise FileNotFoundError(f"not found entity index file: {entity_index_path}")
    if not os.path.exists(relation_index_path):
        raise FileNotFoundError(f"not found relation index file: {relation_index_path}")
    if not os.path.exists(timestamp_index_path):
        raise FileNotFoundError(f"not found timestamp index file: {timestamp_index_path}")


    g = ig.Graph.Read_Pickle(graph_path)
    ent_index = faiss.read_index(entity_index_path)
    rel_index = faiss.read_index(relation_index_path)
    ts_index = faiss.read_index(timestamp_index_path)

    return g, ent_index, rel_index, ts_index, meta



def build_all_with_faiss(
    base_store_path: str,
    normalize: bool = True, 
):

    global_rag_config = RAGConfig()

    txt_path = global_rag_config.get_dataset_kg()
    graph_out_path = f"{base_store_path}/{global_rag_config.DATASET_NAME}.pkl"
    index_out_dir = f"{base_store_path}/index"
    csv_out_dir = f"{base_store_path}/csv"
    meta_file_path = f"{base_store_path}/meta.json"

    if FileHelper.judge_file_exist(meta_file_path):
        logger.info(f"meta file already exists, skipping build: {meta_file_path}")
        return

    for dir_path in [index_out_dir, csv_out_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    g, _ = build_igraph_from_txt(txt_path)

 
    entity_texts = list(g.vs["name"])  
    relation_texts = list(g["relation_vocab"]) 
    ts_texts = list(g["timestamp_vocab"])


    logger.info("Embedding entities ...")
    ent_emb = embed_fn(
        entity_texts, convert_to_numpy=True, normalize=normalize
    )  # [E, d]
    logger.info("Embedding relations ...")
    rel_emb = embed_fn(
        relation_texts, convert_to_numpy=True, normalize=normalize
    )  # [R, d]
    logger.info("Embedding timestamps ...")
    ts_emb = embed_fn(ts_texts, convert_to_numpy=True, normalize=normalize)  # [T, d]


    ent_ids = np.arange(g.vcount(), dtype=np.int64) 
    rel_ids = np.arange(len(relation_texts), dtype=np.int64)  
    ts_ids = np.arange(len(ts_texts), dtype=np.int64)  

    use_cosine = bool(normalize)  
    ent_index = _build_faiss_index(ent_emb, use_cosine, ent_ids)
    rel_index = _build_faiss_index(rel_emb, use_cosine, rel_ids)
    ts_index = _build_faiss_index(ts_emb, use_cosine, ts_ids)

    graph_path = graph_out_path
    ent_idx_path = os.path.join(index_out_dir, "entity.index")
    rel_idx_path = os.path.join(index_out_dir, "relation.index")
    ts_idx_path = os.path.join(index_out_dir, "timestamp.index")

    g.write_pickle(graph_path)
    faiss.write_index(ent_index, ent_idx_path)
    faiss.write_index(rel_index, rel_idx_path)
    faiss.write_index(ts_index, ts_idx_path)


    pd.DataFrame(
        {
            "vid": ent_ids,
            "name": g.vs["name"],
            "node_id": g.vs["node_id"],
        }
    ).to_csv(os.path.join(csv_out_dir, "entity_index.csv"), index=False)

    pd.DataFrame(
        {
            "relation_id": rel_ids,
            "relation": relation_texts,
        }
    ).to_csv(os.path.join(csv_out_dir, "relation_index.csv"), index=False)

    pd.DataFrame(
        {
            "timestamp_id": ts_ids,
            "timestamp": ts_texts,
        }
    ).to_csv(os.path.join(csv_out_dir, "timestamp_index.csv"), index=False)


    dim = ent_emb.shape[1]
    meta = {
        "embedding_dim": int(dim),
        "normalize": bool(normalize),
        "metric": "cosine_via_inner_product" if use_cosine else "l2",
        "counts": {
            "entities": int(g.vcount()),
            "relations": int(len(relation_texts)),
            "timestamps": int(len(ts_texts)),
        },
        "files": {
            "graph": graph_path,
            "entity_index": ent_idx_path,
            "relation_index": rel_idx_path,
            "timestamp_index": ts_idx_path,
            "entity_csv": os.path.join(csv_out_dir, "entity_index.csv"),
            "relation_csv": os.path.join(csv_out_dir, "relation_index.csv"),
            "timestamp_csv": os.path.join(csv_out_dir, "timestamp_index.csv"),
        },
    }
    FileHelper.save_json(meta_file_path, meta)

    logger.info(f"Done. Artifacts saved under: {base_store_path}")
    return {
        "graph": g,
        "ent_index": ent_index,
        "rel_index": rel_index,
        "ts_index": ts_index,
        "ent_emb": ent_emb,
        "rel_emb": rel_emb,
        "ts_emb": ts_emb,
        "meta": meta,
    }


def find_triples_by_three_queries(
    head_query: Optional[str],
    relation_query: Optional[str],
    tail_query: Optional[str],
    graph: ig.Graph,
    ent_index: faiss.Index,
    rel_index: faiss.Index,
    entity_threshold: float = 0.8,
    relation_threshold: float = 0.8,
    topk_per_query: int = 64,
    normalize: bool = True,
    use_range_search: bool = False,
) -> List[Dict[str, Any]]:

    HIGH_CONF_TRIGGER = 0.9
    HIGH_CONF_KEEP = 0.9

    def _search_single(
        index: faiss.Index, query: str, threshold: float, is_entity: bool
    ) -> Dict[int, float]:
        q_emb = embed_fn([query], convert_to_numpy=True, normalize=normalize)

        if use_range_search and hasattr(index, "range_search"):
            lims, D, I = index.range_search(q_emb, threshold)
            beg, end = lims[0], lims[1]
            return {int(i): float(d) for d, i in zip(D[beg:end], I[beg:end]) if i >= 0}
        else:
            D, I = index.search(q_emb, topk_per_query)
            return {
                int(i): float(d)
                for d, i in zip(D[0], I[0])
                if i >= 0 and float(d) >= threshold
            }

    def _apply_high_conf_prune(
        scores: Dict[int, float],
        is_unknown_slot: bool,
        trigger: float = HIGH_CONF_TRIGGER,
        keep: float = HIGH_CONF_KEEP,
    ) -> Dict[int, float]:

        if is_unknown_slot or not scores:
            return scores

        has_high_conf = any(score > trigger for score in scores.values())
        if not has_high_conf:
            return scores

        pruned = {idx: score for idx, score in scores.items() if score >= keep}
        return pruned if pruned else scores

    anchor_type = "none"

    # ===== head =====
    head_is_unknown = not (head_query and head_query != "?")
    if not head_is_unknown:
        head_scores = _search_single(
            ent_index, head_query, entity_threshold, is_entity=True
        )
        head_scores = _apply_high_conf_prune(
            head_scores,
            is_unknown_slot=False,
        )
    else:
        anchor_type = "head"
        head_scores = {v.index: 1.0 for v in graph.vs}

    # ===== relation =====
    relation_is_unknown = not (relation_query and relation_query != "?")
    if not relation_is_unknown:
        rel_scores = _search_single(
            rel_index, relation_query, relation_threshold, is_entity=False
        )
        rel_scores = _apply_high_conf_prune(
            rel_scores,
            is_unknown_slot=False,
        )
    else:
        anchor_type = "relation"
        rel_scores = {i: 1.0 for i in range(len(graph["relation_vocab"]))}

    # ===== tail =====
    tail_is_unknown = not (tail_query and tail_query != "?")
    if not tail_is_unknown:
        tail_scores = _search_single(
            ent_index, tail_query, entity_threshold, is_entity=True
        )
        tail_scores = _apply_high_conf_prune(
            tail_scores,
            is_unknown_slot=False,
        )
    else:
        anchor_type = "tail"
        tail_scores = {v.index: 1.0 for v in graph.vs}

    if not head_scores or not rel_scores or not tail_scores:
        return []

    head_ids = set(head_scores.keys())
    rel_ids = set(rel_scores.keys())
    tail_ids = set(tail_scores.keys())

    simple_triples: List[Dict[str, Any]] = []


    for hi in head_ids:
        out_eids = graph.incident(hi, mode="OUT")
        if not out_eids:
            continue

        h_sim_base = float(head_scores[hi])

        for eid in out_eids:
            edge = graph.es[eid]
            ri = int(edge["relation_id"])
            if ri not in rel_ids:
                continue

            ti = edge.target
            if ti not in tail_ids:
                continue

            r_sim = float(rel_scores[ri])
            t_sim = float(tail_scores[ti])
            score_sum = h_sim_base + r_sim + t_sim

            timestamp = (
                edge["timestamp"] if "timestamp" in edge.attribute_names() else None
            )

            simple_triples.append(
                {
                    "score_sum": score_sum,
                    "head": graph.vs[hi]["name"],
                    "relation": edge["relation"],
                    "tail": graph.vs[ti]["name"],
                    "anchor_type": anchor_type,
                    "timestamp": timestamp,
                    "h_sim": h_sim_base,
                    "r_sim": r_sim,
                    "t_sim": t_sim,
                }
            )

    simple_triples.sort(key=lambda x: x["score_sum"], reverse=True)
    return simple_triples



def find_triples_by_constraint(
    head_query: Optional[str],
    relation_query: Optional[str],
    tail_query: Optional[str],
    graph: ig.Graph,
    ent_index: faiss.Index,
    rel_index: faiss.Index,
    entity_threshold: float = 0.7,
    relation_threshold: float = 0.7,
    topk_per_query: int = 64,
    normalize: bool = True,
    use_range_search: bool = False,
) -> List[Dict[str, Any]]:

    def _search_single(
        index: faiss.Index, query: str, threshold: float, is_entity: bool
    ) -> Dict[int, float]:
        q_emb = embed_fn([query], convert_to_numpy=True, normalize=normalize)

        if use_range_search and hasattr(index, "range_search"):
            lims, D, I = index.range_search(q_emb, float(threshold))
            beg, end = lims[0], lims[1]
            return {
                int(i): float(d)
                for d, i in zip(D[beg:end], I[beg:end])
                if i >= 0 and float(d) >= float(threshold)
            }
        else:
            D, I = index.search(q_emb, topk_per_query)
            return {
                int(i): float(d)
                for d, i in zip(D[0], I[0])
                if i >= 0 and float(d) >= float(threshold)
            }

    anchor_type = "none"

    # ===== head =====
    head_is_unknown = not (head_query and head_query != "?")
    if not head_is_unknown:
        head_scores = _search_single(
            ent_index, head_query, entity_threshold, is_entity=True
        )
    else:
        anchor_type = "head"
        head_scores = {v.index: 1.0 for v in graph.vs}

    # ===== relation =====
    relation_is_unknown = not (relation_query and relation_query != "?")
    if not relation_is_unknown:
        rel_scores = _search_single(
            rel_index, relation_query, relation_threshold, is_entity=False
        )
    else:
        anchor_type = "relation"
        rel_scores = {i: 1.0 for i in range(len(graph["relation_vocab"]))}

    # ===== tail =====
    tail_is_unknown = not (tail_query and tail_query != "?")
    if not tail_is_unknown:
        tail_scores = _search_single(
            ent_index, tail_query, entity_threshold, is_entity=True
        )
    else:
        anchor_type = "tail"
        tail_scores = {v.index: 1.0 for v in graph.vs}

    if not head_scores or not rel_scores or not tail_scores:
        return []

    head_ids = set(head_scores.keys())
    rel_ids = set(rel_scores.keys())
    tail_ids = set(tail_scores.keys())

    simple_triples: List[Dict[str, Any]] = []


    for hi in head_ids:
        out_eids = graph.incident(hi, mode="OUT")
        if not out_eids:
            continue

        h_sim_base = float(head_scores[hi])

        for eid in out_eids:
            edge = graph.es[eid]
            ri = int(edge["relation_id"])                                               
            if ri not in rel_ids:
                continue

            ti = edge.target
            if ti not in tail_ids:
                continue

            r_sim = float(rel_scores[ri])
            t_sim = float(tail_scores[ti])
            score_sum = h_sim_base + r_sim + t_sim

            timestamp = (
                edge["timestamp"] if "timestamp" in edge.attribute_names() else None
            )

            simple_triples.append(
                {
                    "score_sum": score_sum,
                    "head": graph.vs[hi]["name"],
                    "relation": edge["relation"],
                    "tail": graph.vs[ti]["name"],
                    "anchor_type": anchor_type,
                    "timestamp": timestamp,
                    "h_sim": h_sim_base,
                    "r_sim": r_sim,
                    "t_sim": t_sim,
                }
            )

    simple_triples.sort(key=lambda x: x["score_sum"], reverse=True)
    return simple_triples
