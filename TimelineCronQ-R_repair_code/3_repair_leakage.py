#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean duplicate questions in a final merged dataset.

Usage
-----
python clean_duplicate_questions.py --input all.json

Only `--input` is required.

Outputs
-------
Given:
    /path/to/all.json

The script:
    1. rewrites /path/to/all.json in place;
    2. writes /path/to/all.duplicate_cleaned_report.json.

Cleaning policy
---------------
Duplicate key:
    Exact stripped `question` string.

Policy A: medium / complex duplicates
    If a question appears more than once globally, every medium/complex record
    belonging to that duplicate question group is deleted.

Policy B: simple duplicates
    If a duplicate question group contains >= 2 simple records:
        - merge them into one representative sample;
        - aggregate `answer` into a deduplicated list;
        - aggregate `events` into a deduplicated list;
        - sort `events` by:
              1. event.start_time ascending
              2. event string alphabetically ascending
        - sort `answer` by:
              1. the earliest corresponding event.start_time ascending
              2. answer string alphabetically ascending
          If an answer cannot be aligned back to an event, it is placed after
          aligned answers and sorted alphabetically.
        - remove the old `split`;
        - redistribute all merged-simple samples globally with an exact-ish 8:1:1
          train / validation / test allocation.

    If a duplicate question group contains exactly 1 simple record plus
    medium/complex duplicates:
        - medium/complex records are deleted;
        - that single simple record is retained unchanged.

Untouched
---------
Non-duplicate records remain unchanged.
The script does not rewrite `question`, `answer_type`, `question_type`,
`question_level`, or `temporal_relation` except for the intended aggregation
of simple `answer`/`events` fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# JSON helpers
# ============================================================

def load_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_any(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_records_container(
    obj: Any,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Supports:
      1. top-level list:
            [sample, sample, ...]
      2. top-level dict:
            {"data": [sample, ...]}
            {"samples": [sample, ...]}
            {"records": [sample, ...]}
            {"items": [sample, ...]}
    """
    if isinstance(obj, list):
        return obj, None

    if not isinstance(obj, dict):
        raise TypeError(f"Input JSON must be a list or dict, got {type(obj)}")

    for key in ("data", "samples", "records", "items"):
        if key in obj and isinstance(obj[key], list):
            return obj[key], key

    raise KeyError(
        "Top-level JSON is a dict, but no sample list was found under "
        "one of: data / samples / records / items."
    )


def replace_records_in_container(
    obj: Any,
    records: List[Dict[str, Any]],
    resolved_key: Optional[str],
) -> Any:
    if isinstance(obj, list):
        return records

    if resolved_key is None:
        raise RuntimeError("resolved_key is None for dict input")

    out = dict(obj)
    out[resolved_key] = records
    return out


# ============================================================
# Generic helpers
# ============================================================

def normalize_question_key(record: Dict[str, Any], record_index: int) -> str:
    """
    Exact duplicate key = stripped question text.

    Empty/missing questions are intentionally made unique so that they are not
    collapsed accidentally.
    """
    question = record.get("question")
    if question is None:
        return f"__MISSING_QUESTION__::{record_index}"

    question_text = str(question).strip()
    if not question_text:
        return f"__EMPTY_QUESTION__::{record_index}"

    return question_text


def stable_json_key(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def stable_dedupe(items: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()

    for item in items:
        key = stable_json_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)

    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def largest_remainder_811(n: int) -> Dict[str, int]:
    """
    Allocate n items into train/validation/test approximately as 8:1:1
    using the largest remainder method.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    raw = {
        "train": n * 0.8,
        "validation": n * 0.1,
        "test": n * 0.1,
    }
    floors = {k: int(v) for k, v in raw.items()}
    remaining = n - sum(floors.values())

    tie_order = {"validation": 0, "test": 1, "train": 2}
    ranked = sorted(
        raw.keys(),
        key=lambda k: (raw[k] - floors[k], -tie_order[k]),
        reverse=True,
    )

    for i in range(remaining):
        floors[ranked[i]] += 1

    return floors


def count_splits(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for record in records:
        split = str(record.get("split", "__NO_SPLIT__"))
        counter[split] += 1
    return dict(counter)


def count_levels(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for record in records:
        level = str(record.get("question_level", "__NO_LEVEL__"))
        counter[level] += 1
    return dict(counter)


def remaining_duplicate_group_count(records: List[Dict[str, Any]]) -> int:
    by_q = Counter()
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        key = normalize_question_key(record, idx)
        by_q[key] += 1
    return sum(1 for _, count in by_q.items() if count > 1)


# ============================================================
# Event / answer sorting
# ============================================================

def parse_event_str(event_str: Any) -> Optional[Tuple[str, str, str, str, str]]:
    parts = [x.strip() for x in str(event_str).split("|")]
    if len(parts) != 5:
        return None
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def parse_time_for_sort(t: Any) -> datetime:
    text = str(t).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.max


def event_sort_key(event_str: Any) -> Tuple[datetime, str]:
    """
    Events:
        1. start_time ascending
        2. full event string alphabetically ascending
    """
    parsed = parse_event_str(event_str)
    if parsed is None:
        return datetime.max, str(event_str).casefold()

    _, _, _, start_time, _ = parsed
    return parse_time_for_sort(start_time), str(event_str).casefold()


def event_to_answer(
    event: Tuple[str, str, str, str, str],
    answer_type: Any,
) -> Optional[str]:
    s, _, o, start_time, end_time = event
    answer_type = str(answer_type or "")

    if answer_type == "subject":
        return s

    if answer_type == "object":
        return o

    if answer_type == "timestamp_start":
        return start_time

    if answer_type == "timestamp_end":
        return end_time

    if answer_type == "timestamp_range":
        return f"{start_time} to {end_time}"

    if answer_type == "timestamp":
        if start_time == end_time:
            return start_time
        return f"{start_time} to {end_time}"

    if answer_type == "duration":
        return f"{start_time} to {end_time}"

    return None


def build_answer_start_time_index(
    events: List[Any],
    answer_type: Any,
) -> Dict[str, datetime]:
    """
    Map answer string -> earliest corresponding event.start_time.
    """
    answer_to_start: Dict[str, datetime] = {}

    for event_str in events:
        parsed = parse_event_str(event_str)
        if parsed is None:
            continue

        answer = event_to_answer(parsed, answer_type)
        if answer is None:
            continue

        start_dt = parse_time_for_sort(parsed[3])
        old = answer_to_start.get(answer)
        if old is None or start_dt < old:
            answer_to_start[answer] = start_dt

    return answer_to_start


def answer_sort_key(
    answer: Any,
    answer_to_start: Dict[str, datetime],
) -> Tuple[datetime, str]:
    """
    Answers:
        1. earliest aligned event.start_time ascending
        2. answer string alphabetically ascending
    """
    answer_str = str(answer)
    return answer_to_start.get(answer_str, datetime.max), answer_str.casefold()


# ============================================================
# Simple aggregation
# ============================================================

METADATA_FIELDS_TO_CHECK = (
    "question_level",
    "question_type",
    "answer_type",
    "temporal_relation",
)


def metadata_consistency_flags(simple_records: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Return fields whose values differ across simple records.
    Empty dict means consistent.
    """
    inconsistent: Dict[str, List[Any]] = {}

    for field in METADATA_FIELDS_TO_CHECK:
        values = stable_dedupe(record.get(field) for record in simple_records)
        if len(values) > 1:
            inconsistent[field] = values

    return inconsistent


def merge_simple_records(simple_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge >= 2 simple records into one representative sample.

    Representative metadata comes from the first record.
    answer/events are aggregated, deduplicated, then sorted.
    split is removed here and assigned later in one global 8:1:1 pass.
    """
    if len(simple_records) < 2:
        raise ValueError("merge_simple_records requires at least 2 records")

    merged = dict(simple_records[0])

    merged_answers_raw: List[Any] = []
    merged_events_raw: List[Any] = []

    for record in simple_records:
        merged_answers_raw.extend(to_list(record.get("answer")))
        merged_events_raw.extend(to_list(record.get("events")))

    merged_events = stable_dedupe(merged_events_raw)
    merged_events = sorted(merged_events, key=event_sort_key)

    merged_answers = stable_dedupe(merged_answers_raw)
    answer_to_start = build_answer_start_time_index(
        merged_events,
        merged.get("answer_type"),
    )
    merged_answers = sorted(
        merged_answers,
        key=lambda answer: answer_sort_key(answer, answer_to_start),
    )

    merged["answer"] = merged_answers
    merged["events"] = merged_events
    merged.pop("split", None)

    return merged


# ============================================================
# Main cleaning logic
# ============================================================

def clean_dataset(
    input_obj: Any,
) -> Tuple[Any, Dict[str, Any]]:
    records, resolved_key = get_records_container(input_obj)

    original_count = len(records)
    original_split_counts = count_splits(r for r in records if isinstance(r, dict))
    original_level_counts = count_levels(r for r in records if isinstance(r, dict))

    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        key = normalize_question_key(record, idx)
        groups[key].append((idx, record))

    duplicate_groups = {
        question: members
        for question, members in groups.items()
        if len(members) > 1
    }

    first_index_to_question = {
        min(idx for idx, _ in members): question
        for question, members in duplicate_groups.items()
    }
    all_duplicate_indices = {
        idx
        for members in duplicate_groups.values()
        for idx, _ in members
    }

    stats = Counter()
    stats["original_count"] = original_count
    stats["duplicate_question_groups"] = len(duplicate_groups)
    stats["records_inside_duplicate_groups"] = sum(
        len(members) for members in duplicate_groups.values()
    )

    audit_groups: List[Dict[str, Any]] = []
    output_records: List[Any] = []
    merged_simple_records: List[Dict[str, Any]] = []
    merged_simple_audit_refs: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        if idx not in all_duplicate_indices:
            output_records.append(record)
            continue

        if idx not in first_index_to_question:
            continue

        question = first_index_to_question[idx]
        members = duplicate_groups[question]
        member_records = [r for _, r in members]

        simple_records = [
            r for r in member_records
            if r.get("question_level") == "simple"
        ]
        medium_records = [
            r for r in member_records
            if r.get("question_level") == "medium"
        ]
        complex_records = [
            r for r in member_records
            if r.get("question_level") == "complex"
        ]
        other_level_records = [
            r for r in member_records
            if r.get("question_level") not in {"simple", "medium", "complex"}
        ]

        stats["dropped_medium_records"] += len(medium_records)
        stats["dropped_complex_records"] += len(complex_records)
        stats["dropped_medium_complex_records_total"] += (
            len(medium_records) + len(complex_records)
        )

        audit_entry: Dict[str, Any] = {
            "question": question,
            "duplicate_group_size": len(member_records),
            "ids": [r.get("id") for r in member_records],
            "original_splits": [r.get("split") for r in member_records],
            "level_counts": dict(Counter(str(r.get("question_level")) for r in member_records)),
            "dropped_medium_ids": [r.get("id") for r in medium_records],
            "dropped_complex_ids": [r.get("id") for r in complex_records],
            "action": None,
        }

        if other_level_records:
            output_records.extend(other_level_records)
            stats["preserved_unknown_level_records"] += len(other_level_records)
            audit_entry["preserved_unknown_level_ids"] = [
                r.get("id") for r in other_level_records
            ]

        if len(simple_records) >= 2:
            merged = merge_simple_records(simple_records)
            output_records.append(merged)
            merged_simple_records.append(merged)
            merged_simple_audit_refs.append(audit_entry)

            stats["simple_merge_groups"] += 1
            stats["simple_records_merged_input"] += len(simple_records)
            stats["simple_records_merged_output"] += 1
            stats["simple_records_removed_by_merge"] += len(simple_records) - 1

            inconsistent = metadata_consistency_flags(simple_records)
            if inconsistent:
                stats["simple_merge_groups_with_metadata_inconsistency"] += 1
                audit_entry["metadata_inconsistency"] = inconsistent

            audit_entry["action"] = "merge_simple_records_sort_answers_events_and_reassign_split"
            audit_entry["retained_or_merged_id"] = merged.get("id")
            audit_entry["merged_answer_count"] = len(merged.get("answer") or [])
            audit_entry["merged_events_count"] = len(merged.get("events") or [])
            audit_entry["merged_answers"] = merged.get("answer") or []
            audit_entry["merged_events"] = merged.get("events") or []

        elif len(simple_records) == 1:
            singleton = dict(simple_records[0])
            output_records.append(singleton)

            stats["simple_singleton_survivors_in_mixed_duplicate_groups"] += 1

            audit_entry["action"] = "drop_medium_complex_and_keep_single_simple"
            audit_entry["retained_or_merged_id"] = singleton.get("id")

        else:
            stats["duplicate_groups_fully_deleted_without_simple"] += 1
            audit_entry["action"] = "drop_duplicate_group_without_simple"

        audit_groups.append(audit_entry)

    merged_sorted = sorted(
        merged_simple_records,
        key=lambda r: sha256_text(str(r.get("question", "")).strip()),
    )
    allocation = largest_remainder_811(len(merged_sorted))

    split_sequence = (
        ["train"] * allocation["train"]
        + ["validation"] * allocation["validation"]
        + ["test"] * allocation["test"]
    )

    audit_ref_by_question = {
        entry["question"]: entry
        for entry in merged_simple_audit_refs
    }

    reassigned_counter = Counter()
    for record, new_split in zip(merged_sorted, split_sequence):
        record["split"] = new_split
        reassigned_counter[new_split] += 1

        question = str(record.get("question", "")).strip()
        audit_entry = audit_ref_by_question.get(question)
        if audit_entry is not None:
            audit_entry["reassigned_split"] = new_split

    stats["merged_simple_reassigned_train"] = reassigned_counter["train"]
    stats["merged_simple_reassigned_validation"] = reassigned_counter["validation"]
    stats["merged_simple_reassigned_test"] = reassigned_counter["test"]

    final_records: List[Dict[str, Any]] = [
        r for r in output_records if isinstance(r, dict)
    ]
    final_count = len(output_records)

    report: Dict[str, Any] = {
        "summary": {
            **dict(stats),
            "final_count": final_count,
            "net_removed_records": original_count - final_count,
            "remaining_duplicate_question_groups_after_cleaning": remaining_duplicate_group_count(final_records),
            "original_split_counts": original_split_counts,
            "final_split_counts": count_splits(final_records),
            "original_level_counts": original_level_counts,
            "final_level_counts": count_levels(final_records),
            "merged_simple_split_allocation_target": allocation,
        },
        # "duplicate_group_audit": audit_groups,
    }

    cleaned_obj = replace_records_in_container(input_obj, output_records, resolved_key)
    return cleaned_obj, report


# ============================================================
# CLI
# ============================================================

def build_report_path(input_path: Path) -> Path:
    stem = input_path.stem
    return input_path.with_name(f"{stem}.duplicate_cleaned_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete globally duplicated medium/complex questions, "
            "merge duplicated simple questions, sort merged answer/events, "
            "redistribute merged simple samples with an 8:1:1 split, "
            "and rewrite the input JSON in place."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input merged dataset JSON path. This file will be rewritten in place.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    report_path = build_report_path(input_path)

    input_obj = load_json_any(input_path)
    cleaned_obj, report = clean_dataset(input_obj)

    save_json_any(cleaned_obj, input_path)
    save_json_any(report, report_path)

    summary = report["summary"]

    print("=" * 92)
    print("[Duplicate Question Cleaning Report]")
    print(f"Rewritten input: {input_path}")
    print(f"Audit report:    {report_path}")
    print("-" * 92)
    print(f"Original count: {summary['original_count']}")
    print(f"Final count:    {summary['final_count']}")
    print(f"Net removed:    {summary['net_removed_records']}")
    print(f"Duplicate question groups found: {summary['duplicate_question_groups']}")
    print(f"Dropped medium records: {summary['dropped_medium_records']}")
    print(f"Dropped complex records: {summary['dropped_complex_records']}")
    print(f"Simple merge groups: {summary['simple_merge_groups']}")
    print(
        "Merged simple reassigned splits: "
        f"train={summary['merged_simple_reassigned_train']}, "
        f"validation={summary['merged_simple_reassigned_validation']}, "
        f"test={summary['merged_simple_reassigned_test']}"
    )
    print(
        "Remaining duplicate question groups after cleaning: "
        f"{summary['remaining_duplicate_question_groups_after_cleaning']}"
    )
    print("=" * 92)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())