from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from src.config.base_config import logger
from src.utils.dataset_utils import find_triples_by_constraint
from src.utils.temporal_utils import apply_temporal_constraint, sort_candidates_by_time
from src.llm.intent_constraint import run_temporal_constraint_parse

from src.utils.temporal_utils import (
    normalize_timestamp,
    normalize_explicit_interval,
    _match_with_offset,
    _match_without_offset,
)


def _build_exact_lookup(graph) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:

    ent_name_to_ids: Dict[str, List[int]] = defaultdict(list)
    for v in graph.vs:
        ent_name_to_ids[str(v["name"])].append(v.index)

    rel_name_to_ids: Dict[str, List[int]] = defaultdict(list)
    relation_vocab = graph["relation_vocab"]
    for rid, rname in enumerate(relation_vocab):
        rel_name_to_ids[str(rname)].append(rid)

    return ent_name_to_ids, rel_name_to_ids


def _infer_anchor_type(head_query, relation_query, tail_query) -> str:
    anchor_type = "none"
    if head_query == "?":
        anchor_type = "head"
    if relation_query == "?":
        anchor_type = "relation"
    if tail_query == "?":
        anchor_type = "tail"
    return anchor_type


def _direct_exact_search(
    head_query: Optional[str],
    relation_query: Optional[str],
    tail_query: Optional[str],
    graph,
    ent_name_to_ids: Dict[str, List[int]],
    rel_name_to_ids: Dict[str, List[int]],
) -> Optional[List[Dict[str, Any]]]:

    anchor_type = _infer_anchor_type(head_query, relation_query, tail_query)

    if head_query is not None and head_query != "?":
        if head_query not in ent_name_to_ids:
            return None
        head_ids = set(ent_name_to_ids[head_query])
    else:
        head_ids = set(v.index for v in graph.vs)

    if relation_query is not None and relation_query != "?":
        if relation_query not in rel_name_to_ids:
            return None
        rel_ids = set(rel_name_to_ids[relation_query])
    else:
        rel_ids = set(range(len(graph["relation_vocab"])))

    if tail_query is not None and tail_query != "?":
        if tail_query not in ent_name_to_ids:
            return None
        tail_ids = set(ent_name_to_ids[tail_query])
    else:
        tail_ids = set(v.index for v in graph.vs)

    results: List[Dict[str, Any]] = []

    for hi in head_ids:
        out_eids = graph.incident(hi, mode="OUT")
        if not out_eids:
            continue

        for eid in out_eids:
            edge = graph.es[eid]
            ri = int(edge["relation_id"])
            if ri not in rel_ids:
                continue

            ti = edge.target
            if ti not in tail_ids:
                continue

            timestamp = (
                edge["timestamp"] if "timestamp" in edge.attribute_names() else None
            )

            results.append(
                {
                    "score_sum": 3.0,
                    "head": graph.vs[hi]["name"],
                    "relation": edge["relation"],
                    "tail": graph.vs[ti]["name"],
                    "anchor_type": anchor_type,
                    "timestamp": timestamp,
                    "h_sim": 1.0,
                    "r_sim": 1.0,
                    "t_sim": 1.0,
                }
            )

    results.sort(
        key=lambda x: (
            x["timestamp"] is None,
            x["timestamp"] if x["timestamp"] is not None else "",
        )
    )
    return results


def _search_with_fallback(
    head_query: Optional[str],
    relation_query: Optional[str],
    tail_query: Optional[str],
    graph,
    ent_index,
    rel_index,
    ent_name_to_ids: Dict[str, List[int]],
    rel_name_to_ids: Dict[str, List[int]],
    low_entity_threshold: float,
    low_relation_threshold: float,
    topk_per_query: int = 64,
    force_direct_first: bool = True,
) -> List[Dict[str, Any]]:

    if force_direct_first:
        direct_res = _direct_exact_search(
            head_query=head_query,
            relation_query=relation_query,
            tail_query=tail_query,
            graph=graph,
            ent_name_to_ids=ent_name_to_ids,
            rel_name_to_ids=rel_name_to_ids,
        )
        if direct_res:
            return direct_res

    approx_res = find_triples_by_constraint(
        head_query=head_query,
        relation_query=relation_query,
        tail_query=tail_query,
        graph=graph,
        ent_index=ent_index,
        rel_index=rel_index,
        entity_threshold=low_entity_threshold,
        relation_threshold=low_relation_threshold,
        topk_per_query=topk_per_query,
    )
    if approx_res:

        approx_res = approx_res[:50]
        return approx_res

    return []


def apply_ranking_to_candidates(
    candidates: List[Dict[str, Any]],
    ranking: Optional[Dict[str, Any]],
    keep_top_n_for_edge_rank: int = 3,
) -> List[Dict[str, Any]]:
    """
    Apply temporal ranking to candidates.

    ranking format:
        {
            "rank": "asc" | "desc",
            "rank_k": int
        }

    Semantics:
    - rank='asc':
        sort candidates from earliest to latest.
    - rank='desc':
        sort candidates from latest to earliest.
    - rank_k:
        keep candidates from position 1 to position rank_k, inclusive.
    - Special case:
        if rank_k == 1, keep top-N candidates instead of only one candidate,
        where N = keep_top_n_for_edge_rank.
    """

    if not candidates or ranking is None:
        return candidates

    rank = ranking.get("rank")
    rank_k = ranking.get("rank_k")

    if rank not in {"asc", "desc"}:
        raise ValueError(f"Invalid ranking.rank: {rank}")

    if not isinstance(rank_k, int) or rank_k <= 0:
        raise ValueError(f"Invalid ranking.rank_k: {rank_k}")

    sorted_candidates = sort_candidates_by_time(
        candidates,
        reverse=(rank == "desc"),
    )

    if not sorted_candidates:
        return []

    if rank_k == 1:
        keep_n = keep_top_n_for_edge_rank
    else:
        keep_n = rank_k

    return sorted_candidates[:keep_n]


def constrained_retriver(
    graph,
    ent_index,
    rel_index,
    anchor_triples: List[List[str]],
    search_triple: Optional[List[str]] = None,
    constraints: Optional[List[Dict[str, Any]]] = None,
    ranking: Optional[Dict[str, Any]] = None,
    low_entity_threshold: float = 0.5,
    low_relation_threshold: float = 0.5,
    topk_per_query: int = 64,
) -> Tuple[Dict[str, Any], bool]:

    constraints = constraints or []

    def _safe_ratio(before_count: int, after_count: int) -> Optional[float]:
        if before_count is None or after_count is None:
            return None
        if before_count <= 0:
            return None
        return 1.0 - (after_count / before_count)

    def _count_parseable_time(candidates: List[Dict[str, Any]]) -> int:
        count = 0
        for item in candidates:
            if normalize_timestamp(item.get("timestamp")) is not None:
                count += 1
        return count

    ent_name_to_ids, rel_name_to_ids = _build_exact_lookup(graph)

    anchor_results: List[Dict[str, Any]] = []
    anchor_candidate_lists: List[List[Dict[str, Any]]] = []

    evidence_stats: Dict[str, Any] = {
        "has_search_triple": search_triple is not None,
        "num_constraints": len(constraints),
        "anchor_retrieval": [],
        "search_retrieval": None,
        "constraint_steps": [],
        "executed_constraint_count": 0,
        "num_candidates_before_constraints": None,
        "num_candidates_after_constraints": None,
        "constraint_removed_count": None,
        "constraint_reduction_ratio": None,
        "empty_after_constraints": None,
        "ranking_step": {
            "ranking_requested": ranking is not None,
            "ranking_applied": False,
            "ranking": ranking,
            "before_count": None,
            "after_count": None,
            "removed_count": None,
            "reduction_ratio": None,
        },
        "num_candidates_after_ranking_before_truncation": None,
        "num_candidates_returned_to_answer_model": None,
        "empty_after_ranking": None,
        "final_removed_count_from_retrieval": None,
        "final_reduction_ratio_from_retrieval": None,
    }

    for anchor_index, triple in enumerate(anchor_triples):
        simplified = _search_with_fallback(
            head_query=triple[0],
            relation_query=triple[1],
            tail_query=triple[2],
            graph=graph,
            ent_index=ent_index,
            rel_index=rel_index,
            ent_name_to_ids=ent_name_to_ids,
            rel_name_to_ids=rel_name_to_ids,
            low_entity_threshold=low_entity_threshold,
            low_relation_threshold=low_relation_threshold,
            topk_per_query=topk_per_query,
            force_direct_first=True,
        )

        anchor_candidate_lists.append(simplified)
        anchor_results.append(
            {
                "query_triple": triple,
                "query_answer": simplified,
            }
        )

        evidence_stats["anchor_retrieval"].append(
            {
                "anchor_index": anchor_index,
                "query_triple": triple,
                "candidate_count": len(simplified),
                "parseable_time_count": _count_parseable_time(simplified),
                "has_candidates": len(simplified) > 0,
            }
        )

    if search_triple is None:

        total_anchor_returned = 0
        candidate_num = 20
        for idx, anchor_item in enumerate(anchor_results):
            query_answer = anchor_item.get("query_answer", [])
            before_truncation = len(query_answer)

            truncated_answer = query_answer[:candidate_num]
            after_truncation = len(truncated_answer)

            anchor_item["query_answer"] = truncated_answer
            total_anchor_returned += after_truncation

            if idx < len(evidence_stats["anchor_retrieval"]):
                evidence_stats["anchor_retrieval"][idx][
                    "returned_to_answer_model_count"
                ] = after_truncation
                evidence_stats["anchor_retrieval"][idx][
                    "truncated_for_answer_context"
                ] = (before_truncation > after_truncation)

        evidence_stats["anchor_only_topk_per_anchor"] = candidate_num
        evidence_stats["anchor_only_total_facts_returned"] = total_anchor_returned

        has_resolvable_anchor_time = any(
            item["parseable_time_count"] > 0
            for item in evidence_stats["anchor_retrieval"]
        )

        appear_problem = not has_resolvable_anchor_time

        return {
            "anchor_results": anchor_results,
            "evidence_stats": evidence_stats,
        }, appear_problem

    search_candidates = _search_with_fallback(
        head_query=search_triple[0],
        relation_query=search_triple[1],
        tail_query=search_triple[2],
        graph=graph,
        ent_index=ent_index,
        rel_index=rel_index,
        ent_name_to_ids=ent_name_to_ids,
        rel_name_to_ids=rel_name_to_ids,
        low_entity_threshold=low_entity_threshold,
        low_relation_threshold=low_relation_threshold,
        topk_per_query=topk_per_query,
        force_direct_first=True,
    )

    evidence_stats["search_retrieval"] = {
        "query_triple": search_triple,
        "candidate_count": len(search_candidates),
        "parseable_time_count": _count_parseable_time(search_candidates),
        "has_candidates": len(search_candidates) > 0,
    }

    evidence_stats["num_candidates_before_constraints"] = len(search_candidates)

    filtered = search_candidates

    for constraint_index, constraint in enumerate(constraints):
        before_count = len(filtered)

        anchor_kind = constraint.get("anchor_kind")
        relation = constraint.get("relation")
        anchor_index = constraint.get("anchor_index")
        granularity = constraint.get("granularity")
        offset_days = constraint.get("offset_days")

        anchor_candidate_count = None
        anchor_parseable_time_count = None

        if anchor_kind == "event":
            if anchor_index is None:
                raise ValueError("event lacks anchor_index")

            if anchor_index < 0 or anchor_index >= len(anchor_candidate_lists):
                raise IndexError(
                    f"anchor_index out of bounds: {anchor_index}, needed 0~{len(anchor_candidate_lists)-1}"
                )

            selected_anchor_candidates = anchor_candidate_lists[anchor_index]
            anchor_candidate_count = len(selected_anchor_candidates)
            anchor_parseable_time_count = _count_parseable_time(
                selected_anchor_candidates
            )

            filtered = apply_temporal_constraint(
                candidates=filtered,
                constraint=constraint,
                anchor_candidates=selected_anchor_candidates,
            )
        else:
            filtered = apply_temporal_constraint(
                candidates=filtered,
                constraint=constraint,
                anchor_candidates=None,
            )

        after_count = len(filtered)

        evidence_stats["constraint_steps"].append(
            {
                "constraint_index": constraint_index,
                "anchor_kind": anchor_kind,
                "relation": relation,
                "anchor_index": anchor_index,
                "granularity": granularity,
                "offset_days": offset_days,
                "before_count": before_count,
                "after_count": after_count,
                "removed_count": before_count - after_count,
                "reduction_ratio": _safe_ratio(before_count, after_count),
                "anchor_candidate_count": anchor_candidate_count,
                "anchor_parseable_time_count": anchor_parseable_time_count,
                "became_empty": before_count > 0 and after_count == 0,
            }
        )

        if not filtered:
            break

    evidence_stats["executed_constraint_count"] = len(
        evidence_stats["constraint_steps"]
    )

    num_before_constraints = evidence_stats["num_candidates_before_constraints"]
    num_after_constraints = len(filtered)

    evidence_stats["num_candidates_after_constraints"] = num_after_constraints
    evidence_stats["constraint_removed_count"] = (
        num_before_constraints - num_after_constraints
        if num_before_constraints is not None
        else None
    )
    evidence_stats["constraint_reduction_ratio"] = _safe_ratio(
        num_before_constraints,
        num_after_constraints,
    )
    evidence_stats["empty_after_constraints"] = num_after_constraints == 0

    ranking_before_count = len(filtered)

    if filtered and ranking is not None:
        filtered = apply_ranking_to_candidates(
            candidates=filtered,
            ranking=ranking,
            keep_top_n_for_edge_rank=3,
        )
        evidence_stats["ranking_step"]["ranking_applied"] = True

    ranking_after_count = len(filtered)

    evidence_stats["ranking_step"]["before_count"] = ranking_before_count
    evidence_stats["ranking_step"]["after_count"] = ranking_after_count
    evidence_stats["ranking_step"]["removed_count"] = (
        ranking_before_count - ranking_after_count
    )
    evidence_stats["ranking_step"]["reduction_ratio"] = _safe_ratio(
        ranking_before_count,
        ranking_after_count,
    )

    evidence_stats["num_candidates_after_ranking_before_truncation"] = (
        ranking_after_count
    )

    evidence_stats["num_candidates_returned_to_answer_model"] = len(filtered[:20])
    evidence_stats["empty_after_ranking"] = ranking_after_count == 0

    final_count = evidence_stats["num_candidates_returned_to_answer_model"]

    evidence_stats["final_removed_count_from_retrieval"] = (
        num_before_constraints - final_count
        if num_before_constraints is not None and final_count is not None
        else None
    )
    evidence_stats["final_reduction_ratio_from_retrieval"] = _safe_ratio(
        num_before_constraints,
        final_count,
    )

    appear_problem = not filtered

    return {
        "anchor_results": anchor_results,
        "search_result": {
            "query_triple": search_triple,
            "query_answer": filtered[:20],
        },
        "evidence_stats": evidence_stats,
    }, appear_problem


def intent_constraint(aligned):
    question = aligned.get("question", "")
    aligned_triples = aligned.get(
        "aligned_triples",
        {"anchor_triples": [], "search_triple": None},
    )

    anchor_triples = aligned_triples.get("anchor_triples", [])
    search_triple = aligned_triples.get("search_triple", None)

    if search_triple is None:
        return {
            "question": question,
            "anchor_triples": anchor_triples,
            "search_triple": None,
            "constraints": [],
            "ranking": None,
        }

    parsed = run_temporal_constraint_parse(
        question=question,
        decomposed_triples=aligned_triples,
    )

    return {
        "question": question,
        "anchor_triples": anchor_triples,
        "search_triple": search_triple,
        "constraints": [c.model_dump() for c in parsed.constraints],
        "ranking": parsed.ranking.model_dump() if parsed.ranking is not None else None,
    }
