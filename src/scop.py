import os
import re
import shutil
import copy
import random
from typing import List, Dict, Any

import faiss
import igraph as ig
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import openai

from src.config.base_config import logger
from src.config.llm_config import LLMConfig
from src.config.rag_config import RAGConfig, PipelineStatus, TemporalDatasets
from src.utils.common_utils import FileHelper
from src.eval.constraint_eval import run_unified_evaluation

from src.utils.dataset_utils import (
    find_triples_by_three_queries,
    find_triples_by_constraint,
)

from src.co_retriver import (
    align_entity,
    decompose_single_question,
    final_answer,
    format_query_results,
)

from src.constraint import (
    intent_constraint,
    constrained_retriver,
)

from src.eval.eval_multi import final_eval as multi_final_eval
from src.eval.eval_corn import final_eval as corn_final_eval

# ================= 自定义异常 =================


class WorkerTimeoutError(Exception):


    def __init__(self, fallback_data):
        self.fallback_data = fallback_data


class PipelineAbortError(Exception):


    pass


# ================= 通用工具 =================


def ensure_stage(d: Dict[str, Any]) -> Dict[str, Any]:

    if "stage" not in d or not isinstance(d["stage"], dict):
        d["stage"] = {
            "status": {},
            "record": {},
        }

    d["stage"].setdefault("status", {})
    d["stage"].setdefault("record", {})

    for key in ["decompose", "align", "constraint", "answer"]:
        d["stage"]["status"].setdefault(key, PipelineStatus.PENDING)
        d["stage"]["record"].setdefault(key, "")

    return d


def extract_stratified_test_set(
    data: List[Dict[str, Any]],
    test_size: int,
    stratify_keys: List[str] = ["qlabel", "answer_type"],
) -> List[Dict[str, Any]]:
    total_data_len = len(data)
    if total_data_len == 0 or test_size <= 0:
        return []

    actual_test_size = min(test_size, total_data_len)
    grouped_data = defaultdict(list)

    for item in data:
        composite_key = tuple(item.get(k, "Unknown") for k in stratify_keys)
        grouped_data[composite_key].append(item)

    sampled_test_set = []

    for composite_key, items in grouped_data.items():
        proportion = len(items) / total_data_len
        sample_count = round(proportion * actual_test_size)
        sample_count = min(sample_count, len(items))

        if sample_count > 0:
            sampled_test_set.extend(random.sample(items, sample_count))

    diff = actual_test_size - len(sampled_test_set)

    if diff > 0:
        sampled_ids = {id(item) for item in sampled_test_set}
        remaining_data = [item for item in data if id(item) not in sampled_ids]
        sampled_test_set.extend(
            random.sample(remaining_data, min(diff, len(remaining_data)))
        )
    elif diff < 0:
        sampled_test_set = random.sample(sampled_test_set, actual_test_size)

    random.shuffle(sampled_test_set)
    return sampled_test_set


def dedup_facts(facts: List[Dict[str, Any]], topk: int = 50) -> List[Dict[str, Any]]:

    seen = set()
    out = []

    for item in facts:
        key = (
            item.get("head"),
            item.get("relation"),
            item.get("tail"),
            item.get("timestamp"),
            item.get("start_time"),
            item.get("end_time"),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

        if len(out) >= topk:
            break

    return out


def format_flat_facts(facts: List[Dict[str, Any]], max_items: int = 50) -> str:

    lines = []
    seen = set()

    for item in facts:
        h = item.get("head")
        r = item.get("relation")
        t = item.get("tail")

        timestamp = item.get("timestamp")
        start_time = item.get("start_time") or item.get("start")
        end_time = item.get("end_time") or item.get("end")

        key = (h, r, t, timestamp, start_time, end_time)
        if key in seen:
            continue
        seen.add(key)

        if start_time is not None or end_time is not None:
            lines.append(f"({h}, {r}, {t}, start={start_time}, end={end_time})")
        else:
            lines.append(f"({h}, {r}, {t}, time={timestamp})")

        if len(lines) >= max_items:
            break

    return "\n".join(lines)


def atomic_save_json(path: str, data):

    tmp_path = f"{path}.tmp"
    FileHelper.save_json(tmp_path, data)
    os.replace(tmp_path, path)



def decompose_worker(data):
    new_data = dict(data)
    new_data["stage"] = copy.deepcopy(data.get("stage"))
    ensure_stage(new_data)

    STAGE_KEY = "decompose"

    try:
        result = decompose_single_question(new_data)

        if not result.anchor_triples and not result.search_triple:
            new_data["extract_triples"] = {
                "anchor_triples": [],
                "search_triple": None,
            }
            new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SKIPPED
            new_data["stage"]["record"][STAGE_KEY] = "Empty output from model"
            return new_data

        new_data["extract_triples"] = {
            "anchor_triples": [list(t) for t in result.anchor_triples],
            "search_triple": (
                list(result.search_triple) if result.search_triple else None
            ),
        }

        if new_data["stage"]["status"][STAGE_KEY] == PipelineStatus.PENDING:
            new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SUCCESS
            new_data["stage"]["record"][STAGE_KEY] = "Success"

        return new_data

    except openai.APITimeoutError:
        new_data["extract_triples"] = {
            "anchor_triples": [],
            "search_triple": None,
        }
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SKIPPED
        new_data["stage"]["record"][STAGE_KEY] = "Skipped due to API Timeout"
        raise WorkerTimeoutError(new_data)

    except Exception as e:
        logger.warning(f"exception in decompose_worker: {e}", exc_info=True)
        new_data["extract_triples"] = {
            "anchor_triples": [],
            "search_triple": None,
        }
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SKIPPED
        new_data["stage"]["record"][STAGE_KEY] = f"Fatal: {str(e)[:30]}"
        return new_data


def align_worker(decomposed_data, g, ent_index, rel_index):
    new_data = dict(decomposed_data)
    new_data["stage"] = copy.deepcopy(decomposed_data.get("stage"))
    ensure_stage(new_data)

    STAGE_KEY = "align"

    upstream_status = new_data["stage"]["status"].get("decompose")
    if upstream_status in [PipelineStatus.SKIPPED]:
        new_data["aligned_triples"] = {
            "anchor_triples": [],
            "search_triple": None,
        }
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SKIPPED
        new_data["stage"]["record"][
            STAGE_KEY
        ] = f"Skipped due to upstream decompose: {upstream_status}"
        return new_data

    try:
        source_triples = copy.deepcopy(
            decomposed_data.get(
                "extract_triples",
                {"anchor_triples": [], "search_triple": None},
            )
        )
        source_triples["question"] = decomposed_data.get("question", "")

        align_result, appear_problem = align_entity(
            source_triples=source_triples,
            g=g,
            ent_index=ent_index,
            rel_index=rel_index,
            topk_per_query=64,
        )

        new_data["aligned_triples"] = {
            "anchor_triples": [
                list(t) for t in align_result.grounded_triples.anchor_triples
            ],
            "search_triple": (
                list(align_result.grounded_triples.search_triple)
                if align_result.grounded_triples.search_triple is not None
                else None
            ),
        }

        if appear_problem:
            new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.WARNING
            new_data["stage"]["record"][
                STAGE_KEY
            ] = "Alignment had issues, not found enough candidate triples"
            return new_data

        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SUCCESS
        new_data["stage"]["record"][STAGE_KEY] = "Success"
        return new_data

    except openai.APITimeoutError:
        new_data["aligned_triples"] = {
            "anchor_triples": [],
            "search_triple": None,
        }
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.WARNING
        new_data["stage"]["record"][STAGE_KEY] = "Skipped due to API Timeout"
        raise WorkerTimeoutError(new_data)

    except Exception as e:
        new_data["aligned_triples"] = {
            "anchor_triples": [],
            "search_triple": None,
        }
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.WARNING
        new_data["stage"]["record"][STAGE_KEY] = f"Fatal: {str(e)[:30]}"
        return new_data



def constraint_worker(aligned_data, g, ent_index, rel_index):
    new_data = dict(aligned_data)
    new_data["stage"] = copy.deepcopy(aligned_data.get("stage"))
    STAGE_KEY = "constraint"

    status_dict = new_data["stage"]["status"]
    if status_dict.get("align") in [PipelineStatus.SKIPPED]:
        new_data["constraint_plan"] = None
        new_data["ground_graph"] = None
        new_data["constraint_stats"] = None
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SKIPPED
        new_data["stage"]["record"][
            STAGE_KEY
        ] = "Skipped due to upstream align failure/skip"
        return new_data

    try:
        plan = intent_constraint(new_data)

        ground_graph, appear_problem = constrained_retriver(
            graph=g,
            ent_index=ent_index,
            rel_index=rel_index,
            anchor_triples=plan["anchor_triples"],
            search_triple=plan["search_triple"],
            constraints=plan.get("constraints", []),
            ranking=plan.get("ranking"),
            low_entity_threshold=0.5,
            low_relation_threshold=0.5,
        )

        new_data["constraint_plan"] = plan
        new_data["ground_graph"] = ground_graph
        new_data["constraint_stats"] = (
            ground_graph.get("evidence_stats")
            if isinstance(ground_graph, dict)
            else None
        )

        if appear_problem:
            new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.WARNING
            new_data["stage"]["record"][
                STAGE_KEY
            ] = "Filtered data is empty after applying constraints"
        elif plan["search_triple"] and not plan["constraints"]:
            new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.WARNING
            new_data["stage"]["record"][STAGE_KEY] = (
                "Search triple is present but no constraints applied, "
                "which may lead to noisy retrieval results"
            )
        else:
            new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SUCCESS
            new_data["stage"]["record"][STAGE_KEY] = "Success"

        return new_data

    except openai.APITimeoutError as e:
        new_data["constraint_plan"] = None
        new_data["ground_graph"] = None
        new_data["constraint_stats"] = None
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.WARNING
        new_data["stage"]["record"][STAGE_KEY] = "Skipped due to API Timeout"
        raise WorkerTimeoutError(new_data)

    except Exception as e:
        new_data["constraint_plan"] = None
        new_data["ground_graph"] = None
        new_data["constraint_stats"] = None
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.WARNING
        new_data["stage"]["record"][STAGE_KEY] = f"Fatal: {str(e)[:30]}"
        return new_data


def answer_worker(constrained_data):
    new_data = dict(constrained_data)
    new_data["stage"] = copy.deepcopy(constrained_data.get("stage"))
    ensure_stage(new_data)

    STAGE_KEY = "answer"

    status_dict = new_data["stage"]["status"]
    if status_dict.get("constraint") in [PipelineStatus.SKIPPED]:
        new_data["predict_answer"] = []
        new_data["predict_thought"] = None
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SKIPPED
        new_data["stage"]["record"][
            STAGE_KEY
        ] = "Skipped due to upstream constraint failure/skip"
        return new_data

    question = new_data.get("question", "")
    ground_graph = new_data.get("ground_graph")

    try:
        answer_context = new_data.get("answer_context")
        if answer_context is None:
            answer_context = format_query_results(ground_graph)

        thought_answer = final_answer(
            question,
            answer_context,
        )

        new_data["predict_answer"] = thought_answer.answer
        new_data["predict_thought"] = getattr(thought_answer, "thought", None)
        new_data["answer_context"] = answer_context

        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SUCCESS
        new_data["stage"]["record"][STAGE_KEY] = "Success"

        return new_data

    except openai.APITimeoutError:
        logger.warning("问答生成阶段发生超时")
        new_data["predict_answer"] = None
        new_data["predict_thought"] = None
        new_data["answer_context"] = None
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.WARNING
        new_data["stage"]["record"][STAGE_KEY] = "Skipped due to API Timeout"
        raise WorkerTimeoutError(new_data)

    except Exception as e:
        logger.warning(f"问答阶段发生异常: {e}", exc_info=True)
        new_data["predict_answer"] = None
        new_data["predict_thought"] = None
        new_data["answer_context"] = None
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.WARNING
        new_data["stage"]["record"][STAGE_KEY] = f"Fatal: {str(e)[:30]}"
        return new_data


# ================= Ablation Workers =================

def dedup_triples_only(
    facts: List[Dict[str, Any]],
    topk: int = 3,
    collapse_reverse: bool = True,
) -> List[Dict[str, Any]]:

    best_by_key = {}

    for item in facts:
        h = item.get("head")
        r = item.get("relation")
        t = item.get("tail")

        if not h or not r or not t:
            continue

        if collapse_reverse:
            ent_pair = tuple(sorted([h, t]))
            key = (ent_pair[0], r, ent_pair[1])
        else:
            key = (h, r, t)

        score = float(item.get("score_sum", 0.0))

        if key not in best_by_key or score > float(
            best_by_key[key].get("score_sum", 0.0)
        ):
            best_by_key[key] = item

    out = sorted(
        best_by_key.values(),
        key=lambda x: float(x.get("score_sum", 0.0)),
        reverse=True,
    )

    return out[:topk]


def no_triple_decompose_worker(data, g, ent_index, rel_index, candidate_k: int = 3):
    """
    Ablation: w/o Triple Identification.
    Does not invoke the LLM for anchor/search event parsing.
    Hybrid weak substitution strategy:
    1. Perform weak three-slot retrieval using the original question, taking the top-k unique triples as pseudo anchors;
    2. Use [question, question, question] as the pseudo search_triple;
    3. Subsequently, proceed to the alignment, constraint, and answer stages.
    """
    new_data = dict(data)
    new_data["stage"] = copy.deepcopy(data.get("stage"))
    new_data["ablation_type"] = "no_triple"
    ensure_stage(new_data)

    question = new_data.get("question", "")
    STAGE_KEY = "decompose"

    try:
        weak_triples = find_triples_by_three_queries(
            head_query=question,
            relation_query=question,
            tail_query=question,
            graph=g,
            ent_index=ent_index,
            rel_index=rel_index,
            entity_threshold=0.3,
            relation_threshold=0.3,
            topk_per_query=64,
        )

        weak_triples = dedup_triples_only(
            weak_triples,
            topk=candidate_k,
            collapse_reverse=False,
        )

        pseudo_anchor_triples = [
            [item["head"], item["relation"], item["tail"]]
            for item in weak_triples
            if item.get("head") and item.get("relation") and item.get("tail")
        ]

        pseudo_search_triple = [question, question, question]

        new_data["extract_triples"] = {
            "anchor_triples": pseudo_anchor_triples,
            "search_triple": pseudo_search_triple,
        }

        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.SUCCESS
        new_data["stage"]["record"][STAGE_KEY] = (
            "Ablation no_triple: replace structured triple identification "
            "with hybrid weak retrieval; "
            f"pseudo_anchor_count={len(pseudo_anchor_triples)}, "
            "pseudo_search_triple=[question, question, question]"
        )

        new_data["no_triple_debug"] = {
            "weak_anchor_count": len(pseudo_anchor_triples),
            "weak_anchor_triples": pseudo_anchor_triples,
            "pseudo_search_triple": pseudo_search_triple,
        }

        return new_data

    except Exception as e:
        logger.warning(f"no_triple pseudo decomposition failed: {e}", exc_info=True)

        new_data["extract_triples"] = {
            "anchor_triples": [],
            "search_triple": [question, question, question] if question else None,
        }
        new_data["stage"]["status"][STAGE_KEY] = PipelineStatus.WARNING
        new_data["stage"]["record"][STAGE_KEY] = f"Fatal: {str(e)[:50]}"

        return new_data


def skip_align_worker(decomposed_data):

    new_data = dict(decomposed_data)
    new_data["stage"] = copy.deepcopy(decomposed_data.get("stage"))
    ensure_stage(new_data)

    source = copy.deepcopy(
        decomposed_data.get(
            "extract_triples",
            {"anchor_triples": [], "search_triple": None},
        )
    )

    new_data["aligned_triples"] = {
        "anchor_triples": copy.deepcopy(source.get("anchor_triples", [])),
        "search_triple": copy.deepcopy(source.get("search_triple")),
    }

    new_data["stage"]["status"]["align"] = PipelineStatus.SUCCESS
    new_data["stage"]["record"][
        "align"
    ] = "Ablation no_align: use raw extracted triples as aligned triples"

    return new_data


def no_constraint_worker(aligned_data, g, ent_index, rel_index, topk: int = 10):

    new_data = dict(aligned_data)
    new_data["stage"] = copy.deepcopy(aligned_data.get("stage"))
    ensure_stage(new_data)

    aligned_triples = new_data.get(
        "aligned_triples",
        {"anchor_triples": [], "search_triple": None},
    )
    question = new_data.get("question", "")

    try:
        all_facts = []

        search_triple = aligned_triples.get("search_triple")
        anchor_triples = aligned_triples.get("anchor_triples", [])

        targets = []

        if search_triple:
            targets.append(search_triple)
        else:
            targets.extend(anchor_triples or [])

        if not targets:

            facts = find_triples_by_three_queries(
                head_query=question,
                relation_query=question,
                tail_query=question,
                graph=g,
                ent_index=ent_index,
                rel_index=rel_index,
                entity_threshold=0.1,
                relation_threshold=0.1,
                topk_per_query=64,
            )
            all_facts.extend(facts)
        else:
            for tri in targets:
                if not tri or len(tri) != 3:
                    continue

                h, r, t = tri

                facts = find_triples_by_constraint(
                    head_query=h,
                    relation_query=r,
                    tail_query=t,
                    graph=g,
                    ent_index=ent_index,
                    rel_index=rel_index,
                    entity_threshold=0.5,
                    relation_threshold=0.5,
                    topk_per_query=64,
                )
                all_facts.extend(facts)

        all_facts = dedup_facts(all_facts, topk=topk)

        new_data["constraint_plan"] = {
            "anchor_triples": anchor_triples,
            "search_triple": search_triple,
            "constraints": [],
            "ranking": None,
            "ablation": "no_constraint",
        }
        # new_data["ground_graph"] = None
        # new_data["all_facts"] = all_facts
        new_data["answer_context"] = format_flat_facts(all_facts, max_items=topk)


        new_data["stage"]["status"]["constraint"] = PipelineStatus.SUCCESS
        new_data["stage"]["record"][
            "constraint"
        ] = "Ablation no_constraint: similarity retrieval without constraint execution"

        return new_data

    except Exception as e:
        logger.warning(f"no_constraint retrieval failed: {e}", exc_info=True)

        new_data["constraint_plan"] = None
        new_data["ground_graph"] = None
        new_data["answer_context"] = ""

        new_data["stage"]["status"]["constraint"] = PipelineStatus.WARNING
        new_data["stage"]["record"]["constraint"] = f"Fatal: {str(e)[:50]}"

        return new_data


# ================= 数据清洗 =================


def clean_final_data(new_data):
    logger.info("-> cleaning final data...")

    for d in new_data:
        d.pop("ground_graph", None)
        d.pop("answer_context", None)

        extract_triples = d.get("extract_triples")
        if extract_triples and isinstance(extract_triples, dict):
            extract_triples.pop("question", None)

        plan = d.get("constraint_plan")
        if plan and isinstance(plan, dict):
            plan.pop("question", None)
            plan.pop("anchor_triples", None)
            plan.pop("search_triple", None)

    return new_data




def run_once(
    run_id,
    ablation_root,
    test_size,
    g,
    ent_index,
    rel_index,
    ablation_type="full",
    ablation_topk=10,
    max_timeouts=20,
    shard_size = 500
):
    """
    ablation_type:
      - full
      - no_triple
      - no_align
      - no_constraint
    """
    if ablation_type not in ["full", "no_triple", "no_align", "no_constraint"]:
        raise ValueError(
            f"Unsupported ablation_type: {ablation_type}. "
            f"Expected one of full/no_triple/no_align/no_constraint."
        )

    logger.info(f"====== Run {run_id} START | ablation_type={ablation_type} ======")
    
    # shard_size = 2500
    
    global_llm_config = LLMConfig()
    llm_type = global_llm_config.get_model_name()
    base_url = global_llm_config.get_base_url()
    max_workers = global_llm_config.MAX_WORKERS
    dataset_name = RAGConfig().DATASET_NAME

    logger.info(f"Using model: {llm_type}，url：{base_url}")
    common_dir = os.path.join(ablation_root, "common")
    run_dir = os.path.join(
        ablation_root, ablation_type, f"test_{test_size}", f"run_{run_id}"
    )

    os.makedirs(common_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    common_sample_file = os.path.join(common_dir, f"test_{test_size}_run_{run_id}.json")

    paths = {
        "source": RAGConfig().get_dataset_test_file(),
        "sample": common_sample_file,

        "decompose": os.path.join(run_dir, "q_decomposed.json"),
        "decompose_ckpt_dir": os.path.join(run_dir, "q_decomposed_ckpt"),

        "align": os.path.join(run_dir, "aligned_decomposed.json"),
        "align_ckpt_dir": os.path.join(run_dir, "aligned_decomposed_ckpt"),

        "constraint": os.path.join(run_dir, "constrained.json"),
        "constraint_ckpt_dir": os.path.join(run_dir, "constrained_ckpt"),

        "answer": os.path.join(run_dir, "answer_file.json"),
        "answer_ckpt_dir": os.path.join(run_dir, "answer_file_ckpt"),
    }

    def _run_parallel(
        func,
        data,
        desc,
        checkpoint_dir=None,
        shard_size: int = 2500,
    ) -> List[Dict]:

        timeout_count = 0
        results = [None] * len(data)

        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")

        shard_file_pattern = re.compile(r"^part_(\d+)\.json$")

        def _get_shard_files():
            if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
                return []

            shard_files = []
            for filename in os.listdir(checkpoint_dir):
                match = shard_file_pattern.match(filename)
                if not match:
                    continue

                shard_idx = int(match.group(1))
                shard_path = os.path.join(checkpoint_dir, filename)
                shard_files.append((shard_idx, shard_path))

            shard_files.sort(key=lambda x: x[0])
            return shard_files

        def _load_shards():
            shard_files = _get_shard_files()
            restored_count = 0
            max_shard_idx = 0

            if not shard_files:
                return restored_count, max_shard_idx

            logger.info(
                f"[{desc}] detected shard checkpoint directory, starting recovery: {checkpoint_dir}"
            )

            for shard_idx, shard_path in shard_files:
                max_shard_idx = max(max_shard_idx, shard_idx)

                try:
                    shard_items = FileHelper.load_json(shard_path)
                except Exception as e:
                    logger.warning(
                        f"[{desc}] shard loading failed, skipping file: {shard_path}, error={e}",
                        exc_info=True,
                    )
                    continue

                if not isinstance(shard_items, list):
                    logger.warning(
                        f"[{desc}]  shard structure is invalid, expected list, skipping file: {shard_path}"
                    )
                    continue

                for item in shard_items:
                    if not isinstance(item, dict):
                        continue

                    idx = item.get("idx")
                    item_data = item.get("data")

                    if not isinstance(idx, int):
                        continue
                    if idx < 0 or idx >= len(results):
                        continue
                    if item_data is None:
                        continue

                    if results[idx] is None:
                        restored_count += 1

                    results[idx] = item_data

            logger.info(
                f"[{desc}] shard checkpoint recovery completed, {restored_count}/{len(results)} items restored."
            )
            return restored_count, max_shard_idx

        def _save_shard(shard_items, shard_idx):
            if not checkpoint_dir or not shard_items:
                return

            os.makedirs(checkpoint_dir, exist_ok=True)
            shard_path = os.path.join(checkpoint_dir, f"part_{shard_idx:06d}.json")
            atomic_save_json(shard_path, shard_items)

            logger.info(
                f"[{desc}] shard checkpoint saved: {shard_path}, this shard has {len(shard_items)} items."
            )

        restored_count, max_shard_idx = _load_shards()
        next_shard_idx = max_shard_idx + 1

        pending_indices = [
            i for i, result in enumerate(results)
            if result is None
        ]

        if not pending_indices:
            logger.info(f"[{desc}] shard checkpoint shows current stage is complete, returning directly.")
            return results

        logger.info(
            f"[{desc}] stage recovery status: "
            f"completed {restored_count}/{len(data)}, "
            f"still need to process {len(pending_indices)} items."
        )

        shard_buffer = []

        def _flush_shard_buffer():
            nonlocal shard_buffer, next_shard_idx
            if not shard_buffer:
                return

            _save_shard(shard_buffer, next_shard_idx)
            next_shard_idx += 1
            shard_buffer = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {
                ex.submit(func, data[i]): i
                for i in pending_indices
            }

            for future in tqdm(
                as_completed(future_to_idx),
                total=len(data),
                initial=restored_count,
                desc=f"{desc}(thread {max_workers})",
                ncols=100,
            ):
                idx = future_to_idx[future]

                try:
                    results[idx] = future.result()

                except WorkerTimeoutError as e:
                    timeout_count += 1
                    logger.error(
                        f"[{desc}] timeout occurred! current累计: "
                        f"{timeout_count}/{max_timeouts}"
                    )
                    results[idx] = e.fallback_data

                    if timeout_count >= max_timeouts:
                        logger.critical(
                            f"[{desc}] stage timeout reached {max_timeouts} times, task failed!"
                        )

                        shard_buffer.append(
                            {
                                "idx": idx,
                                "data": results[idx],
                            }
                        )
                        _flush_shard_buffer()

                        for f in future_to_idx:
                            f.cancel()
                        raise PipelineAbortError(
                            f"[{desc}] stage failed due to API timeout threshold reached ({max_timeouts}), aborting remaining tasks."
                        )

                except Exception as e:
                    logger.error(f"[{desc}] thread-level crash: {e}", exc_info=True)
                    results[idx] = data[idx]

                shard_buffer.append(
                    {
                        "idx": idx,
                        "data": results[idx],
                    }
                )

                if len(shard_buffer) >= shard_size:
                    _flush_shard_buffer()

        _flush_shard_buffer()

        return results

    try:

        if FileHelper.judge_file_exist(paths["sample"]):
            logger.info(f"shared test set exists, loading... current test_size={test_size}: {paths['sample']}")
            current_data = FileHelper.load_json(paths["sample"])
        else:
            logger.info("-> Executing Stage 0: Generating shared test set file")
            test_data = FileHelper.load_json(paths["source"])

            if dataset_name == TemporalDatasets.MULTITQ.value:
                current_data = extract_stratified_test_set(
                    data=test_data,
                    test_size=test_size,
                    stratify_keys=["qlabel", "answer_type"],
                )
            else:
                current_data = extract_stratified_test_set(
                    data=test_data,
                    test_size=test_size,
                    stratify_keys=["question_level"],
                )

            for d in current_data:
                ensure_stage(d)

            FileHelper.save_json(paths["sample"], current_data)

        for d in current_data:
            ensure_stage(d)

        if FileHelper.judge_file_exist(paths["answer"]):
            logger.info(f"answer_file exists, skipping recalculation: {paths['answer']}")

        else:

            if FileHelper.judge_file_exist(paths["decompose"]):
                logger.info(f"-> Loading existing decomposition results: {paths['decompose']}")
                current_data = FileHelper.load_json(paths["decompose"])
            else:
                if ablation_type == "no_triple":
                    logger.info(
                        "-> Ablation no_triple: skipping LLM triple identification and using original question for weak triple retrieval"
                    )
                    current_data = _run_parallel(
                        lambda d: no_triple_decompose_worker(
                            d,
                            g,
                            ent_index,
                            rel_index,
                            candidate_k=3,
                        ),
                        current_data,
                        "No-triple pseudo decomposition",
                        checkpoint_dir=paths["decompose_ckpt_dir"],
                        shard_size=shard_size,
                    )
                else:
                    logger.info("-> Executing Stage 1: Question Decomposition")
                    current_data = _run_parallel(
                        lambda d: decompose_worker(d),
                        current_data,
                        "Decompose Question",
                        checkpoint_dir=paths["decompose_ckpt_dir"],
                        shard_size=shard_size,
                    )

                FileHelper.save_json(paths["decompose"], current_data)
                if os.path.isdir(paths["decompose_ckpt_dir"]):
                    shutil.rmtree(paths["decompose_ckpt_dir"])

            if FileHelper.judge_file_exist(paths["align"]):
                logger.info(f"-> Loading existing alignment results: {paths['align']}")
                current_data = FileHelper.load_json(paths["align"])
            else:
                if ablation_type == "no_align":
                    logger.info(
                        "-> Ablation no_align: skipping conservative graph alignment"
                    )
                    current_data = _run_parallel(
                        skip_align_worker,
                        current_data,
                        "跳过对齐",
                        checkpoint_dir=paths["align_ckpt_dir"],
                        shard_size=shard_size,
                    )
                else:
                    logger.info("-> Executing Stage 2: Entity Alignment")
                    current_data = _run_parallel(
                        lambda d: align_worker(
                            d,
                            g,
                            ent_index,
                            rel_index,
                        ),
                        current_data,
                        "Entity Alignment",
                        checkpoint_dir=paths["align_ckpt_dir"],
                        shard_size=shard_size,
                    )

                FileHelper.save_json(paths["align"], current_data)
                if os.path.isdir(paths["align_ckpt_dir"]):
                    shutil.rmtree(paths["align_ckpt_dir"])

            if FileHelper.judge_file_exist(paths["constraint"]):
                logger.info(f"-> Loading existing constraint/retrieval results: {paths['constraint']}")
                current_data = FileHelper.load_json(paths["constraint"])
            else:
                if ablation_type == "no_constraint":
                    logger.info(
                        "-> Ablation no_constraint: " "skipping constraint execution, using similarity-based retrieval"
                    )
                    current_data = _run_parallel(
                        lambda d: no_constraint_worker(
                            d,
                            g,
                            ent_index,
                            rel_index,
                            topk=ablation_topk,
                        ),
                        current_data,
                        "No-constraint Similarity Retrieval",
                        checkpoint_dir=paths["constraint_ckpt_dir"],
                        shard_size=shard_size,
                    )
                else:
                    logger.info("-> Executing Stage 3: Constraint Space")
                    current_data = _run_parallel(
                        lambda d: constraint_worker(
                            d,
                            g,
                            ent_index,
                            rel_index,
                        ),
                        current_data,
                        "Constraint Space",
                        checkpoint_dir=paths["constraint_ckpt_dir"],
                        shard_size=shard_size,
                    )

                FileHelper.save_json(paths["constraint"], current_data)
                if os.path.isdir(paths["constraint_ckpt_dir"]):
                    shutil.rmtree(paths["constraint_ckpt_dir"])


            logger.info("-> Executing Stage 4: Model Answer")
            current_data = _run_parallel(
                lambda d: answer_worker(d),
                current_data,
                "Model Answer",
                checkpoint_dir=paths["answer_ckpt_dir"],
                shard_size=shard_size,
            )

            current_data = clean_final_data(current_data)
            FileHelper.save_json(paths["answer"], current_data)
            if os.path.isdir(paths["answer_ckpt_dir"]):
                shutil.rmtree(paths["answer_ckpt_dir"])

    except PipelineAbortError as e:
        logger.error(f"====== Run {run_id} 被强行终止：{e} ======")
        return

    if dataset_name == TemporalDatasets.MULTITQ.value:
        multi_final_eval(paths["answer"])
    else:
        corn_final_eval(paths["answer"])

    run_unified_evaluation(paths["constraint"],dataset_name)
    
    logger.info(
        f"====== Run {run_id} FINISHED | ablation_type={ablation_type} ======\n"
    )
