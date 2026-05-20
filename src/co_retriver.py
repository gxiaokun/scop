import re
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from src.config.base_config import BaseConfig, logger
from src.config.rag_config import RAGConfig, PipelineStatus
from src.utils.dataset_utils import (
    find_triples_by_three_queries,
    find_triples_by_constraint,
)
from src.utils.llm_utils import embed_fn
from src.llm.triples_extract import run_extraction, ExtractionSchema
from src.llm.entity_align import run_entity_align, AlignSchema, GroundedTriplesSchema
from src.llm.intent_constraint import (
    run_temporal_constraint_parse,
    TemporalConstraintSchema,
)
from src.llm.final_qa import run_final_qa, QASchema


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\[\]\{\}]", "", text).lower().strip()


def repair_search_orientation_by_anchors(
    anchor_triples: List[List[str]],
    search_triple: Optional[List[str]],
    rel_sim_threshold: float = 0.9,
) -> Optional[List[str]]:

    if search_triple is None:
        return search_triple

    if len(search_triple) != 3:
        return search_triple

    q_positions = [i for i, x in enumerate(search_triple) if str(x).strip() == "?"]
    if len(q_positions) != 1:
        return search_triple

    unknown_idx = q_positions[0]
    if unknown_idx not in (0, 2):
        return search_triple

    known_idx = 2 if unknown_idx == 0 else 0

    if str(search_triple[1]).strip() == "?":
        return search_triple

    known_entity = normalize_text(search_triple[known_idx])
    search_relation_raw = str(search_triple[1])
    search_relation_norm = normalize_text(search_relation_raw)

    if not known_entity:
        return search_triple

    entity_matched = []
    for tri in anchor_triples:
        if tri is None or len(tri) != 3:
            continue

        ah, ar, at = tri

        if str(ah).strip() == "?" or str(at).strip() == "?":
            continue

        ah_norm = normalize_text(ah)
        at_norm = normalize_text(at)

        if known_idx == 0:

            if ah_norm == known_entity:
                entity_matched.append(("same", tri))
            elif at_norm == known_entity:
                entity_matched.append(("flip", tri))
        else:

            if at_norm == known_entity:
                entity_matched.append(("same", tri))
            elif ah_norm == known_entity:
                entity_matched.append(("flip", tri))

    if not entity_matched:
        return search_triple

    comparable_sides: List[str] = []
    non_exact_relations: List[str] = []
    non_exact_side_buf: List[str] = []

    for side, tri in entity_matched:
        anchor_rel_raw = str(tri[1])
        anchor_rel_norm = normalize_text(anchor_rel_raw)

        if anchor_rel_norm == search_relation_norm:
            comparable_sides.append(side)
        else:
            non_exact_relations.append(anchor_rel_raw)
            non_exact_side_buf.append(side)

    if non_exact_relations:
        texts = [search_relation_raw] + non_exact_relations
        embs = embed_fn(texts, convert_to_numpy=True, normalize=True)
        search_vec = embs[0]
        anchor_vecs = embs[1:]
        sims = anchor_vecs @ search_vec

        for side, sim in zip(non_exact_side_buf, sims):
            if float(sim) > rel_sim_threshold:
                comparable_sides.append(side)

    if not comparable_sides:
        return search_triple

    same_cnt = sum(1 for s in comparable_sides if s == "same")
    flip_cnt = sum(1 for s in comparable_sides if s == "flip")

    if flip_cnt > 0 and same_cnt == 0:
        repaired = [search_triple[2], search_triple[1], search_triple[0]]

        return repaired

    return search_triple


def _strip_and_dedup_triples(triples_list: List[dict]) -> List[dict]:

    if not triples_list:
        return []

    seen_signatures = set()
    cleaned_list = []

    for tdict in triples_list:
        h_val = tdict.get("head", "")
        r_val = tdict.get("relation", "")
        t_val = tdict.get("tail", "")

        sig = (h_val, r_val, t_val)

        if sig not in seen_signatures:
            seen_signatures.add(sig)

            cleaned_list.append(
                {
                    "head": h_val,
                    "relation": r_val,
                    "tail": t_val,
                    "anchor_type": tdict.get("anchor_type"),
                }
            )

    return cleaned_list


def align_retrieve_candidates(
    triple: List[str],
    allow_orig_fallback: bool,
    g,
    ent_index,
    rel_index,
    entity_sim,
    relation_sim,
    topk_per_query: int = 64,
    min_sim: float = 0.2,
) -> Tuple[bool, List[dict], bool]:

    has_problem = False

    h, r, t = triple[0], triple[1], triple[2]
    thresholds = [(entity_sim, relation_sim)]

    if entity_sim != min_sim or relation_sim != min_sim:
        thresholds.append((min_sim, min_sim))

    for cur_entity_sim, cur_relation_sim in thresholds:

        simplified_triples = find_triples_by_three_queries(
            h,
            r,
            t,
            g,
            ent_index,
            rel_index,
            cur_entity_sim,
            cur_relation_sim,
            topk_per_query,
        )

        if simplified_triples:
            simplified_triples = _strip_and_dedup_triples(simplified_triples)
            return False, simplified_triples, False

        simplified_triples2 = find_triples_by_three_queries(
            t,
            r,
            h,
            g,
            ent_index,
            rel_index,
            cur_entity_sim,
            cur_relation_sim,
            topk_per_query,
        )

        if simplified_triples2:

            simplified_triples2 = _strip_and_dedup_triples(simplified_triples2)
            return True, simplified_triples2, False

    has_problem = True

    if allow_orig_fallback:
        cd = {
            "head": h,
            "relation": r,
            "tail": t,
            "anchor_type": None,
        }
        return False, [cd], has_problem

    return False, [], has_problem


def align_entity(
    source_triples: dict,
    g,
    ent_index,
    rel_index,
    gold_num=10,
    entity_sim=0.4,
    relation_sim=0.4,
    topk_per_query=64,
) -> Tuple[AlignSchema, bool]:

    def is_unknown(s: str) -> bool:
        return s is None or str(s).strip() == "?"

    def orient_candidate(cd: dict, swapped: bool) -> Tuple[str, str, str]:

        if swapped:
            return (
                cd.get("tail", ""),
                cd.get("relation", ""),
                cd.get("head", ""),
            )
        return (
            cd.get("head", ""),
            cd.get("relation", ""),
            cd.get("tail", ""),
        )

    def _dedup_keep_order(items: List[str]) -> List[str]:
        return list(dict.fromkeys(items))

    def _dedup_keep_order_triples(triples: List[List[str]]) -> List[List[str]]:
        seen = set()
        out = []
        for tri in triples:
            sig = tuple(tri)
            if sig not in seen:
                seen.add(sig)
                out.append(tri)
        return out

    anchor_problem = False
    search_problem = False

    anchor_triples = source_triples.get("anchor_triples", [])
    search_triple_raw = source_triples.get("search_triple", None)
    original_question = source_triples.get("question", "")

    search_triple = None
    if search_triple_raw is not None:
        search_triple = list(search_triple_raw)

    has_search = search_triple is not None
    flat_triples: List[List[str]] = [list(t) for t in anchor_triples]
    if has_search:
        flat_triples.append(search_triple)

    anchor_processed: List[Tuple[List[str], bool, List[dict]]] = []
    anchor_gold_entities_raw: List[str] = []
    anchor_gold_relations_raw: List[str] = []
    anchor_gold_triples_raw: List[List[str]] = []

    for triple in anchor_triples:
        original_triple = list(triple)
        swapped, cand_triples, anchor_problem = align_retrieve_candidates(
            original_triple,
            allow_orig_fallback=True,
            g=g,
            ent_index=ent_index,
            rel_index=rel_index,
            entity_sim=entity_sim,
            relation_sim=relation_sim,
            topk_per_query=topk_per_query,
        )
        anchor_processed.append((original_triple, swapped, cand_triples))

        for tdict in cand_triples:
            gold_h, gold_r, gold_t = orient_candidate(tdict, swapped)
            anchor_gold_triples_raw.append([gold_h, gold_r, gold_t])

            if tdict.get("anchor_type") != "head":
                anchor_gold_entities_raw.append(tdict.get("head", ""))
            if tdict.get("anchor_type") != "tail":
                anchor_gold_entities_raw.append(tdict.get("tail", ""))
            if tdict.get("anchor_type") != "relation":
                anchor_gold_relations_raw.append(tdict.get("relation", ""))

    search_processed: Optional[Tuple[List[str], bool, List[dict]]] = None
    search_gold_entities_raw: List[str] = []
    search_gold_relations_raw: List[str] = []

    if has_search:
        swapped, cand_triples, search_problem = align_retrieve_candidates(
            search_triple,
            allow_orig_fallback=False,
            g=g,
            ent_index=ent_index,
            rel_index=rel_index,
            entity_sim=entity_sim,
            relation_sim=relation_sim,
            topk_per_query=topk_per_query,
        )
        search_processed = (list(search_triple), swapped, cand_triples)

        search_head_known = not is_unknown(search_triple[0])
        search_tail_known = not is_unknown(search_triple[2])

        for tdict in cand_triples:
            gold_h, gold_r, gold_t = orient_candidate(tdict, swapped)

            search_gold_relations_raw.append(gold_r)

            if search_head_known:
                search_gold_entities_raw.append(gold_h)
            if search_tail_known:
                search_gold_entities_raw.append(gold_t)

    full_gold_entities = _dedup_keep_order(
        anchor_gold_entities_raw + search_gold_entities_raw
    )
    full_gold_relations = _dedup_keep_order(
        anchor_gold_relations_raw + search_gold_relations_raw
    )
    full_gold_triples = _dedup_keep_order_triples(anchor_gold_triples_raw)

    new_anchor_triples = [list(t) for t in anchor_triples]
    new_search_triple = list(search_triple) if has_search else None
    unified_pool: List[List[str]] = []

    llm_input = {
        "anchor_triples": new_anchor_triples,
        "search_triple": new_search_triple if has_search else None,
    }

    llm_schema_result = run_entity_align(
        source_triples=llm_input,
        gold_entities=full_gold_entities[:gold_num],
        gold_relations=full_gold_relations[:gold_num],
        gold_triples=full_gold_triples[:gold_num],
        original_question=original_question,
    )

    for t in llm_schema_result.grounded_triples.anchor_triples:
        unified_pool.append(list(t))

    if has_search and llm_schema_result.grounded_triples.search_triple is not None:
        unified_pool.append(list(llm_schema_result.grounded_triples.search_triple))

    num_anchors = len(anchor_triples)
    if has_search and len(unified_pool) > num_anchors:
        repaired_search = repair_search_orientation_by_anchors(
            anchor_triples=unified_pool[:num_anchors],
            search_triple=unified_pool[num_anchors],
            rel_sim_threshold=0.9,
        )
        if repaired_search is not None:
            unified_pool[num_anchors] = repaired_search

    num_anchors = len(anchor_triples)

    formatted_anchor = [
        (str(t[0]), str(t[1]), str(t[2])) for t in unified_pool[:num_anchors]
    ]

    formatted_search = None
    if has_search and len(unified_pool) > num_anchors:
        s = unified_pool[num_anchors]
        formatted_search = (str(s[0]), str(s[1]), str(s[2]))

    return (
        AlignSchema(
            grounded_triples=GroundedTriplesSchema(
                anchor_triples=formatted_anchor,
                search_triple=formatted_search,
            )
        ),
        anchor_problem or search_problem,
    )


def decompose_single_question(query_data) -> ExtractionSchema:
    question = query_data.get("question", "")
    return run_extraction(question)


def final_answer(
    question: str,
    answer_context: dict[str, list[str]],
) -> QASchema:
    try:
        return run_final_qa(question, answer_context)
    except Exception as e:
        return QASchema(thought="", answer=[])


def triple_to_temporal_text(item: Dict[str, Any]) -> str:
    head = str(item.get("head", "")).strip()
    relation = str(item.get("relation", "")).strip()
    tail = str(item.get("tail", "")).strip()
    timestamp = item.get("timestamp", None)

    triple_text = f"[{head}, {relation}, {tail}]"

    if timestamp in (None, "", "None"):
        return triple_text

    ts = str(timestamp).strip()

    m = re.fullmatch(r"(.+?)\s+to\s+(.+)", ts, flags=re.IGNORECASE)
    if m:
        left, right = m.groups()
        return f"{triple_text} from {left.strip()} to {right.strip()}"

    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})", ts)
    if m:
        left, right = m.groups()
        return f"{triple_text} from {left} to {right}"

    m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", ts)
    if m:
        left, right = m.groups()
        return f"{triple_text} from {left} to {right}"

    return f"{triple_text} on {ts}"


def format_query_results(
    query_results: Dict[str, Any],
    deduplicate: bool = True,
    max_anchor_items: Optional[int] = None,
    max_search_items: Optional[int] = None,
) -> Dict[str, List[str]]:
    """
    temporal_constrained_search convert：
    {
        "temporal_anchor": [
            "[A, r, B] on 2001-02-23",
            ...
        ],
        "temporal_search": [
            "[A, r, B] on 2001-02-23",
            ...
        ]
    }

    input：
    {
        "anchor_results": [
            {"query_triple": ..., "query_answer": [...]},
            ...
        ],
        "search_result": {
            "query_triple": ...,
            "query_answer": [...]
        }
    }
    """

    def collect_texts(items: List[Dict[str, Any]]) -> List[str]:
        texts = [triple_to_temporal_text(x) for x in items if isinstance(x, dict)]
        if deduplicate:
            texts = list(dict.fromkeys(texts))
        return texts

    temporal_anchor: List[str] = []
    temporal_search: List[str] = []

    anchor_results = query_results.get("anchor_results", [])
    if isinstance(anchor_results, list):
        for anchor_block in anchor_results:
            if not isinstance(anchor_block, dict):
                continue
            answers = anchor_block.get("query_answer", [])
            if isinstance(answers, list):
                temporal_anchor.extend(collect_texts(answers))

    search_result = query_results.get("search_result", {})
    if isinstance(search_result, dict):
        answers = search_result.get("query_answer", [])
        if isinstance(answers, list):
            temporal_search = collect_texts(answers)

    if deduplicate:
        temporal_anchor = list(dict.fromkeys(temporal_anchor))
        temporal_search = list(dict.fromkeys(temporal_search))

    if max_anchor_items is not None:
        temporal_anchor = temporal_anchor[:max_anchor_items]
    if max_search_items is not None:
        temporal_search = temporal_search[:max_search_items]

    return {
        "temporal_anchor": temporal_anchor,
        "temporal_search": temporal_search,
    }
