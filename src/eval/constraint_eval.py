#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Global Evaluation Analysis for SCoP
(Evidence Compression + Gold Answer Coverage + Gold Event Retention)

This script supports both command-line usage and direct function calls.
The provided evaluation file is treated directly as the target assessment set.

Core evaluation behavior:
* Filters duration answers such as 'xxx days' and pure numeric answers from Gold Answer Coverage denominators.
* Splits candidate interval timestamps to correctly match endpoint answers.
* Ignores relation surface-forms during Gold Event Retention and compares event keys at the entity-time level.
* Allows reverse head/tail matching for event retention when semantically equivalent inverse event formulations are used.
* Preserves strict end-to-end event evaluation: missing triples, unfound targets, and empty candidate pools are scored as failures.
* Extracts YYYY and YYYY-MM-DD safely from candidate timestamps.

Dataset-aware reporting policy:
* MultiTQ:
    - Use `qlabel` as the internal level label source.
* Other datasets or unspecified datasets:
    - Use `question_level` as the internal level label source.

Printed reports:
* Overall Evidence Compression — By Level
* Overall Evidence Compression — By Qtype
* Gold Answer Retention — By Level, only for datasets that support gold answer/event evaluation
* Gold Answer Retention — By Qtype, only for datasets that support gold answer/event evaluation
* Gold Event Retention — By Level, only for datasets that support gold answer/event evaluation
* Gold Event Retention — By Qtype, only for datasets that support gold answer/event evaluation
"""

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from src.config.rag_config import TemporalDatasets



DATE_RE = re.compile(r"\d{4}(?:-\d{2}-\d{2})?")


JsonData = Union[List[Dict[str, Any]], Dict[str, Any]]
EvalSource = Union[str, Path, JsonData]
MetricFunc = Callable[[pd.DataFrame, str], Dict[str, Any]]


def load_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("Input JSON must be either a list or a dict.")


def nested_get(obj: Dict[str, Any], keys: List[str], default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def to_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return None


def normalize_text(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    text = unicodedata.normalize("NFKC", str(x))
    text = re.sub(r"\s+", " ", text.strip())
    return text.casefold()


def display_label(x: Any, default: str = "unknown") -> str:
    if x is None:
        return default
    text = str(x).strip()
    return text if text else default


def safe_percent(numer: float, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return numer / denom * 100.0


def reduction_percent(initial_value: float, final_value: float) -> float:
    if initial_value <= 0:
        return float("nan")
    ratio = 1.0 - final_value / initial_value
    if math.isnan(ratio):
        return float("nan")
    return ratio * 100.0


def round_or_nan(x: float, digits: int = 3):
    if isinstance(x, float) and math.isnan(x):
        return float("nan")
    return round(x, digits)


def is_multitq_dataset(dataset_name: Optional[str]) -> bool:
    return normalize_text(dataset_name) == "multitq" if dataset_name else False


def get_dataset_level_label(
    item: Dict[str, Any],
    dataset_name: Optional[str] = None,
) -> str:

    if is_multitq_dataset(dataset_name):
        return display_label(item.get("qlabel"))
    return display_label(item.get("question_level"))


def get_constraint_stats(item: Dict[str, Any]) -> Dict[str, Any]:
    stats = item.get("constraint_stats")
    if isinstance(stats, dict):
        return stats

    evidence_stats = nested_get(item, ["ground_graph", "evidence_stats"])
    if isinstance(evidence_stats, dict):
        return evidence_stats

    return {}


def infer_has_search_triple(item: Dict[str, Any], stats: Dict[str, Any]) -> Optional[bool]:
    if "has_search_triple" in stats:
        return bool(stats.get("has_search_triple"))

    for path in [
        ["aligned_triples", "search_triple"],
        ["extract_triples", "search_triple"],
        ["constraint_plan", "search_triple"],
    ]:
        triple = nested_get(item, path)
        if isinstance(triple, list) and len(triple) == 3:
            return True

    return False


def get_anchor_retrieval(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = stats.get("anchor_retrieval")
    if isinstance(records, list):
        return [x for x in records if isinstance(x, dict)]
    return []


def extract_search_branch_counts(item: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    initial_value = to_number(stats.get("num_candidates_before_constraints"))
    final_value = to_number(stats.get("num_candidates_returned_to_answer_model"))
    evaluable = initial_value is not None and final_value is not None

    return {
        "mechanism_branch": "search_branch",
        "compression_evaluable": evaluable,
        "initial_evidence": initial_value,
        "final_evidence": final_value,
        "non_evaluable_reason": None if evaluable else "missing search branch counts",
    }


def extract_anchor_only_counts(
    item: Dict[str, Any],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    anchor_records = get_anchor_retrieval(stats)

    if not anchor_records:
        return {
            "mechanism_branch": "anchor_only",
            "compression_evaluable": False,
            "initial_evidence": None,
            "final_evidence": None,
            "non_evaluable_reason": "missing anchor_retrieval",
        }

    initial_values: List[float] = []
    initial_valid = True

    for rec in anchor_records:
        i_val = to_number(rec.get("candidate_count"))
        if i_val is None:
            initial_valid = False
        else:
            initial_values.append(i_val)

    initial_value = sum(initial_values) if initial_valid else None

    final_value = 0
    anchor_results = nested_get(item, ["ground_graph", "anchor_results"])

    if isinstance(anchor_results, list):
        for ar in anchor_results:
            qa = ar.get("query_answer")
            if isinstance(qa, list):
                final_value += len(qa)
    else:
        for rec in anchor_records:
            f_val = to_number(rec.get("returned_to_answer_model_count"))
            if f_val is not None:
                final_value += f_val

    evaluable = initial_value is not None and final_value is not None

    return {
        "mechanism_branch": "anchor_only",
        "compression_evaluable": evaluable,
        "initial_evidence": initial_value,
        "final_evidence": final_value,
        "non_evaluable_reason": None if evaluable else "missing anchor counts",
    }

def extract_overall_evidence_counts(item: Dict[str, Any]) -> Dict[str, Any]:
    stats = get_constraint_stats(item)
    has_search_triple = infer_has_search_triple(item, stats)

    if has_search_triple:
        result = extract_search_branch_counts(item, stats)
    else:
        result = extract_anchor_only_counts(item, stats)

    result["has_search_triple"] = has_search_triple
    result["num_constraints"] = to_number(stats.get("num_constraints"))

    return result

def parse_gold_event(event_str: Any) -> Optional[Tuple[str, str, str, str]]:
    if not isinstance(event_str, str):
        return None

    parts = event_str.split("|")
    if len(parts) != 5:
        return None

    head, _, tail, start, end = [normalize_text(p) for p in parts]
    if not head or not tail or not start or not end:
        return None

    return head, tail, start, end


def candidate_to_event_key(candidate: Dict[str, Any]) -> Optional[Tuple[str, str, str, str]]:
    head = normalize_text(candidate.get("head"))
    tail = normalize_text(candidate.get("tail"))
    timestamp = candidate.get("timestamp")

    if not head or not tail or timestamp is None:
        return None

    dates = DATE_RE.findall(str(timestamp))
    if len(dates) == 1:
        start, end = dates[0], dates[0]
    elif len(dates) >= 2:
        start, end = dates[0], dates[1]
    else:
        return None

    return head, tail, start, end


def event_keys_match_allow_reverse(
    gold_event: Tuple[str, str, str, str],
    candidate_event: Tuple[str, str, str, str],
) -> bool:

    gold_head, gold_tail, gold_start, gold_end = gold_event
    cand_head, cand_tail, cand_start, cand_end = candidate_event

    if gold_start != cand_start or gold_end != cand_end:
        return False

    direct_match = gold_head == cand_head and gold_tail == cand_tail
    reverse_match = gold_head == cand_tail and gold_tail == cand_head

    return direct_match or reverse_match


def gold_event_matches_search_fixed_slots(
    gold_event: Tuple[str, str, str, str],
    query_triple: List[str],
) -> bool:

    gold_head, gold_tail, _, _ = gold_event
    query_head, _, query_tail = query_triple

    qh_unknown = str(query_head).strip() == "?"
    qt_unknown = str(query_tail).strip() == "?"

    qh = normalize_text(query_head)
    qt = normalize_text(query_tail)

    direct_match = (
        (qh_unknown or gold_head == qh)
        and (qt_unknown or gold_tail == qt)
    )

    if direct_match:
        return True

    reverse_match = (
        (qh_unknown or gold_tail == qh)
        and (qt_unknown or gold_head == qt)
    )

    return reverse_match


def get_gold_answers(item: Dict[str, Any]) -> Optional[List[Any]]:

    answers = item.get("answer")
    if isinstance(answers, list) and len(answers) > 0:
        return answers

    answers = item.get("answers")
    if isinstance(answers, list) and len(answers) > 0:
        return answers

    return None


def format_gold_answer(ans_str: str) -> Optional[str]:

    ans_str = ans_str.strip().lower()

    # Exclude numeric-only answers from the graph-based coverage denominator.
    if re.match(r"^\d+$", ans_str):
        return None

    # Exclude duration-style answers because they are derived outputs rather than
    # facts expected to be directly preserved in the final candidate graph.
    if re.match(r"^\d+\s+days?$", ans_str):
        return None

    m_range = re.match(
        r"^(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})$",
        ans_str,
    )
    if m_range:
        return f"{m_range.group(1)}-{m_range.group(2)}"

    m_single = re.match(r"^(\d{4}-\d{2}-\d{2})$", ans_str)
    if m_single:
        return f"{m_single.group(1)}-{m_single.group(1)}"

    return normalize_text(ans_str)


def extract_candidate_timestamps(ts_val: Any) -> Set[str]:
    res: Set[str] = set()
    if not ts_val:
        return res

    ts_str = str(ts_val).strip()

    if len(ts_str) == 10 and re.match(r"^\d{4}-\d{2}-\d{2}$", ts_str):
        res.add(f"{ts_str}-{ts_str}")
    else:
        res.add(normalize_text(ts_str))

    m = re.match(
        r"^(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})$",
        ts_str,
    )
    if m:
        res.add(f"{m.group(1)}-{m.group(1)}")
        res.add(f"{m.group(2)}-{m.group(2)}")

    return res


def get_final_candidates(
    item: Dict[str, Any],
    branch: str,
) -> List[Dict[str, Any]]:
    if branch == "search_branch":
        cands = nested_get(item, ["ground_graph", "search_result", "query_answer"])
        return cands if isinstance(cands, list) else []

    cands: List[Dict[str, Any]] = []
    anchor_results = nested_get(item, ["ground_graph", "anchor_results"])

    if isinstance(anchor_results, list):
        for ar in anchor_results:
            qa = ar.get("query_answer")
            if isinstance(qa, list):
                cands.extend(qa)

    return cands

def get_search_triple(item: Dict[str, Any]) -> Optional[List[str]]:
    for path in [
        ["aligned_triples", "search_triple"],
        ["extract_triples", "search_triple"],
        ["constraint_plan", "search_triple"],
    ]:
        triple = nested_get(item, path)
        if isinstance(triple, list) and len(triple) == 3:
            return triple

    return None


def unknown_slot_from_search_triple(search_triple: Optional[List[str]]) -> Optional[str]:
    if not search_triple or len(search_triple) != 3:
        return None

    head, _, tail = search_triple

    if str(head).strip() == "?":
        return "head"
    if str(tail).strip() == "?":
        return "tail"

    return None


def extract_candidate_answer(candidate: Dict[str, Any], slot: str) -> Optional[str]:
    return candidate.get(slot)


def compute_final_answer_coverage(
    item: Dict[str, Any],
    branch: str,
) -> Dict[str, Any]:
    result = {
        "answer_evaluable": False,
        "num_gold_answers": None,
        "num_retained_gold_answers": None,
        "any_gold_answer_covered": None,
        "all_gold_answers_covered": None,
        "answer_recall": None,
    }

    answers = get_gold_answers(item)
    if not answers:
        return result

    valid_gold_answers: Set[str] = set()
    for ans in answers:
        formatted_ans = format_gold_answer(str(ans))
        if formatted_ans:
            valid_gold_answers.add(formatted_ans)

    valid_gold_answer_list = list(valid_gold_answers)
    if not valid_gold_answer_list:
        return result

    num_gold = len(valid_gold_answer_list)

    def return_failure() -> Dict[str, Any]:
        result.update({
            "answer_evaluable": True,
            "num_gold_answers": num_gold,
            "num_retained_gold_answers": 0,
            "any_gold_answer_covered": False,
            "all_gold_answers_covered": False,
            "answer_recall": 0.0,
        })
        return result

    candidates = get_final_candidates(item, branch)
    if not candidates:
        return return_failure()

    candidate_answers: Set[str] = set()

    if branch == "search_branch":
        search_triple = get_search_triple(item)
        if not search_triple or len(search_triple) != 3:
            return return_failure()

        slot = unknown_slot_from_search_triple(search_triple)

        for cand in candidates:
            if not isinstance(cand, dict):
                continue

            if slot:
                ans = extract_candidate_answer(cand, slot)
                if ans:
                    candidate_answers.add(normalize_text(ans))

            candidate_answers.update(extract_candidate_timestamps(cand.get("timestamp")))

    else:
        for cand in candidates:
            if isinstance(cand, dict):
                cand_h = normalize_text(cand.get("head"))
                cand_t = normalize_text(cand.get("tail"))

                if cand_h:
                    candidate_answers.add(cand_h)
                if cand_t:
                    candidate_answers.add(cand_t)

                candidate_answers.update(extract_candidate_timestamps(cand.get("timestamp")))

    if not candidate_answers:
        return return_failure()

    retained = set(valid_gold_answer_list) & candidate_answers
    retained_count = len(retained)

    result.update({
        "answer_evaluable": True,
        "num_gold_answers": num_gold,
        "num_retained_gold_answers": retained_count,
        "any_gold_answer_covered": retained_count > 0,
        "all_gold_answers_covered": retained_count == num_gold,
        "answer_recall": retained_count / num_gold,
    })

    return result

def compute_final_gold_event_retention(
    item: Dict[str, Any],
    branch: str,
) -> Dict[str, Any]:
    result = {
        "event_evaluable": False,
        "num_gold_search_events": None,
        "num_retained_gold_search_events": None,
        "any_gold_event_retained": None,
        "all_gold_events_retained": None,
        "event_recall": None,
    }

    events = item.get("events")
    if not isinstance(events, list):
        return result

    parsed_gold_events = [e for e in (parse_gold_event(es) for es in events) if e]
    if not parsed_gold_events:
        return result

    def return_failure(
        num_gold: int = len(parsed_gold_events),
    ) -> Dict[str, Any]:
        result.update({
            "event_evaluable": True,
            "num_gold_search_events": num_gold,
            "num_retained_gold_search_events": 0,
            "any_gold_event_retained": False,
            "all_gold_events_retained": False,
            "event_recall": 0.0,
        })
        return result

    gold_search_events: Set[Tuple[str, str, str, str]] = set()

    if branch == "search_branch":
        search_triple = get_search_triple(item)
        if not search_triple or len(search_triple) != 3:
            return return_failure()

        gold_search_events = {
            e for e in parsed_gold_events
            if gold_event_matches_search_fixed_slots(e, search_triple)
        }

    else:
        anchor_triples = nested_get(item, ["aligned_triples", "anchor_triples"])
        if not anchor_triples:
            return return_failure()

        for gold_e in parsed_gold_events:
            for at in anchor_triples:
                if (
                    isinstance(at, list)
                    and len(at) == 3
                    and gold_event_matches_search_fixed_slots(gold_e, at)
                ):
                    gold_search_events.add(gold_e)
                    break

    if not gold_search_events:
        return return_failure()

    actual_num_gold = len(gold_search_events)

    candidates = get_final_candidates(item, branch)
    if not candidates:
        return return_failure(actual_num_gold)

    final_candidate_events = {
        e
        for e in (
            candidate_to_event_key(c)
            for c in candidates
            if isinstance(c, dict)
        )
        if e
    }

    if not final_candidate_events:
        return return_failure(actual_num_gold)

    retained = {
        gold_e
        for gold_e in gold_search_events
        if any(
            event_keys_match_allow_reverse(gold_e, cand_e)
            for cand_e in final_candidate_events
        )
    }
    retained_count = len(retained)

    result.update({
        "event_evaluable": True,
        "num_gold_search_events": actual_num_gold,
        "num_retained_gold_search_events": retained_count,
        "any_gold_event_retained": retained_count > 0,
        "all_gold_events_retained": retained_count == actual_num_gold,
        "event_recall": retained_count / actual_num_gold,
    })

    return result

def build_dataframe(
    data: List[Dict[str, Any]],
    dataset_name: Optional[str] = None,
    compute_gold_metrics: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for item in data:
        count_info = extract_overall_evidence_counts(item)
        branch = count_info["mechanism_branch"]
        initial_value = count_info["initial_evidence"]
        final_value = count_info["final_evidence"]

        sample_reduction_percent = None
        if (
            count_info["compression_evaluable"]
            and initial_value is not None
            and final_value is not None
            and initial_value > 0
        ):
            sample_reduction_percent = (
                1.0 - final_value / initial_value
            ) * 100.0

        row = {
            "id": item.get("id", item.get("quid")),
            "question_level": get_dataset_level_label(item, dataset_name=dataset_name),
            "qtype": display_label(item.get("qtype")),
            "mechanism_branch": branch,
            "has_search_triple": count_info["has_search_triple"],
            "num_constraints": count_info["num_constraints"],
            "compression_evaluable": count_info["compression_evaluable"],
            "initial_evidence": initial_value,
            "final_evidence": final_value,
            "sample_reduction_percent": sample_reduction_percent,
        }

        if compute_gold_metrics:
            row.update(compute_final_answer_coverage(item, branch))
            row.update(compute_final_gold_event_retention(item, branch))

        rows.append(row)

    return pd.DataFrame(rows)

def summarize_compression(df: pd.DataFrame, subset_name: str) -> Dict[str, Any]:
    eval_df = df[df["compression_evaluable"] == True]
    if eval_df.empty:
        return {
            "Subset": subset_name,
            "Evaluable": 0,
            "Initial/Avg": float("nan"),
            "Final/Avg": float("nan"),
            "Compression (%)": float("nan"),
        }

    total_initial = float(eval_df["initial_evidence"].sum())
    total_final = float(eval_df["final_evidence"].sum())

    return {
        "Subset": subset_name,
        "Evaluable": len(eval_df),
        "Initial/Avg": round_or_nan(float(eval_df["initial_evidence"].mean())),
        "Final/Avg": round_or_nan(float(eval_df["final_evidence"].mean())),
        "Compression (%)": round_or_nan(reduction_percent(total_initial, total_final)),
    }


def summarize_answer_coverage(df: pd.DataFrame, subset_name: str) -> Dict[str, Any]:
    eval_df = df[df["answer_evaluable"] == True]
    n = len(eval_df)

    if n == 0:
        return {
            "Subset": subset_name,
            "Evaluable": 0,
            "Any Cov (%)": float("nan"),
            "Full Cov (%)": float("nan"),
            "Mean Recall (%)": float("nan"),
        }

    any_cov = int(eval_df["any_gold_answer_covered"].sum())
    full_cov = int(eval_df["all_gold_answers_covered"].sum())

    return {
        "Subset": subset_name,
        "Evaluable": n,
        "Any Cov (%)": round_or_nan(safe_percent(any_cov, n)),
        "Full Cov (%)": round_or_nan(safe_percent(full_cov, n)),
        "Mean Recall (%)": round_or_nan(float(eval_df["answer_recall"].mean()) * 100.0),
    }


def summarize_event_retention(df: pd.DataFrame, subset_name: str) -> Dict[str, Any]:
    eval_df = df[df["event_evaluable"] == True]
    n = len(eval_df)

    if n == 0:
        return {
            "Subset": subset_name,
            "Evaluable": 0,
            "Any Ret (%)": float("nan"),
            "Full Ret (%)": float("nan"),
            "Mean Recall (%)": float("nan"),
        }

    any_ret = int(eval_df["any_gold_event_retained"].sum())
    full_ret = int(eval_df["all_gold_events_retained"].sum())

    return {
        "Subset": subset_name,
        "Evaluable": n,
        "Any Ret (%)": round_or_nan(safe_percent(any_ret, n)),
        "Full Ret (%)": round_or_nan(safe_percent(full_ret, n)),
        "Mean Recall (%)": round_or_nan(float(eval_df["event_recall"].mean()) * 100.0),
    }


def build_metric_tables(df: pd.DataFrame, func: MetricFunc) -> Dict[str, pd.DataFrame]:
    overall_row = func(df, "Overall")

    by_level: List[Dict[str, Any]] = []
    for level, sub_df in df.groupby("question_level", dropna=False):
        by_level.append(func(sub_df, f"Level: {level}"))

    level_order = {
        "level: simple": 0,
        "level: medium": 1,
        "level: complex": 2,
    }
    by_level.sort(
        key=lambda x: level_order.get(str(x["Subset"]).casefold(), 99)
    )
    by_level.append(overall_row)

    # Preserve qtype order by first occurrence in the filtered dataframe.
    qtype_values = list(dict.fromkeys(df["qtype"].tolist()))
    by_qtype: List[Dict[str, Any]] = []
    for qtype in qtype_values:
        sub_df = df[df["qtype"] == qtype]
        by_qtype.append(func(sub_df, f"Qtype: {qtype}"))
    by_qtype.append(overall_row)

    return {
        "by_level": pd.DataFrame(by_level),
        "by_qtype": pd.DataFrame(by_qtype),
    }



def print_table(
    title: str,
    table: pd.DataFrame,
    note: Optional[str] = None,
) -> None:
    print("=" * 140)
    print(title)
    if note:
        print(note)
    print("=" * 140)
    print(table.to_string(index=False) if not table.empty else "No valid rows.")
    print()


def run_unified_evaluation(
    eval_source: EvalSource,
    dataset_name: Optional[str] = None,
    print_reports: bool = True,
) -> Dict[str, Any]:
    
    if dataset_name not in TemporalDatasets._value2member_map_:
        print(f"ERROR: not supported Dataset[{dataset_name}]")
        return {}

    supports_gold_metrics = not is_multitq_dataset(dataset_name)

    if isinstance(eval_source, (str, Path)):
        data = load_json(eval_source)
        input_source_repr = str(eval_source)
    elif isinstance(eval_source, list):
        data = eval_source
        input_source_repr = "<in-memory list>"
    elif isinstance(eval_source, dict):
        data = [eval_source]
        input_source_repr = "<in-memory dict>"
    else:
        raise TypeError(
            "eval_source must be a file path, a list of dicts, or a single dict."
        )

    df = build_dataframe(
        data=data,
        dataset_name=dataset_name,
        compute_gold_metrics=supports_gold_metrics,
    )

    if df.empty:
        raise ValueError("No samples available for evaluation.")

    comp_tables = build_metric_tables(df, summarize_compression)

    ans_tables = (
        build_metric_tables(df, summarize_answer_coverage)
        if supports_gold_metrics
        else None
    )
    evt_tables = (
        build_metric_tables(df, summarize_event_retention)
        if supports_gold_metrics
        else None
    )

    metadata = {
        "input_source": input_source_repr,
        "dataset_name": dataset_name,
        "normalized_dataset_name": normalize_text(dataset_name) if dataset_name else None,
        "is_multitq": is_multitq_dataset(dataset_name),
        "gold_answer_event_metrics_supported": supports_gold_metrics,
        "num_samples": len(df),
        "compression_evaluable": int(df["compression_evaluable"].fillna(False).sum()),
        "answer_coverage_evaluable": (
            int(df["answer_evaluable"].fillna(False).sum())
            if supports_gold_metrics
            else None
        ),
        "event_retention_evaluable": (
            int(df["event_evaluable"].fillna(False).sum())
            if supports_gold_metrics
            else None
        ),
    }

    if print_reports:
        compression_note = (
            "Note: samples without evaluable evidence counts are skipped, including missing search-branch counts or unavailable anchor retrieval/count statistics."
        )
        answer_coverage_note = (
            "Note: samples without evaluable gold answers are skipped, including missing answer fields or samples whose gold answers are all numeric-only or duration-style. Other retrieval failures are counted as zero coverage."
        )

        print("=" * 140)
        print(f"Total samples included in evaluation: {metadata['num_samples']}")
        print("=" * 140)
        print()

        print_table(
            "Overall Evidence Compression — By Level",
            comp_tables["by_level"],
            note=compression_note,
        )
        print_table(
            "Overall Evidence Compression — By Qtype",
            comp_tables["by_qtype"],
            note=compression_note,
        )
        if supports_gold_metrics:
            print_table(
                "Gold Answer Retention — By Level",
                ans_tables["by_level"],
                note=answer_coverage_note,
            )
            print_table(
                "Gold Answer Retention — By Qtype",
                ans_tables["by_qtype"],
                note=answer_coverage_note,
            )
            print_table(
                "Gold Event Retention — By Level",
                evt_tables["by_level"],
            )
            print_table(
                "Gold Event Retention — By Qtype",
                evt_tables["by_qtype"],
            )

    return {
        "details": df,
        "compression": comp_tables,
        "answer_coverage": ans_tables,
        "event_retention": evt_tables,
        "metadata": metadata,
    }

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        "--eval-file",
        dest="input",
        type=str,
        required=True,
        help="Path to the evaluation JSON file. '--eval-file' is an alias of '--input'.",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="TimelineCronQR",
        help=(
            "Optional dataset name."
        ),
    )

    args = parser.parse_args()

    run_unified_evaluation(
        eval_source=args.input,
        dataset_name=args.dataset_name,
        print_reports=True,
    )


if __name__ == "__main__":
    main()
