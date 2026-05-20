#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final dataset formatting, cleaning, balancing, and resplitting.

Usage
-----
python final_format_final.py --input dataset.json

Optional:
    --output_dir /path/to/output_dir

Outputs
-------
Given:
    dataset.json

The script writes:
    <output_dir>/<input_stem>.final_all.json
    <output_dir>/train.json
    <output_dir>/validation.json
    <output_dir>/test.json
    <output_dir>/<input_stem>.final_format_report.json

Main pipeline
-------------
1. Retain the existing level adjustment rule:
       complex + timeline_position_retrieval*3
       + relation_union_or_intersection + union
       -> medium

2. Normalize answers:
       - "equals" -> "equal"
       - scalar answer -> [answer]
       - list answer remains list
       - strip surrounding whitespace for string answers

3. Drop invalid-answer samples:
       answer == []
       answer == ""
       answer == "Noanswer"
       answer == "No Answer"
       answer == ["No Answer"]
       and equivalent all-invalid variants after normalization.

4. Drop samples with len(answer) > 10.

5. Remove source_kg_id and old split.

6. Sort events:
       - start_time ascending
       - tie-break by full event string alphabetically

7. Sort answers when they can be aligned exactly to event fields:
       - iterate events in sorted order;
       - use subject / relation / object field order inside each event;
       - if every answer item can be found exactly among event s/r/o fields,
         sort answers by that event-field order;
       - if any answer item is not alignable, preserve the normalized answer order.

   This keeps non-entity answers such as:
       ["equal"], [1], ["1"]
   untouched in order.

8. Balance question_level among:
       simple / medium / complex

   Let target = min(count_simple, count_medium, count_complex).
   Each level is reduced to target.

   When a level must be reduced:
       - remove samples from the currently largest question_type bucket first;
       - ties are resolved deterministically;
       - records selected for removal inside a bucket are deterministic by hash.

9. Re-split the final balanced dataset with train / validation / test = 8:1:1.

   Splitting is stratified by:
       (question_level, question_type)

   This approximately preserves:
       - final question_level distribution
       - final question_type distribution

10. Output one overall file and three split files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# Constants
# ============================================================

TARGET_LEVELS: Tuple[str, str, str] = ("simple", "medium", "complex")
SPLIT_NAMES: Tuple[str, str, str] = ("train", "validation", "test")
SPLIT_WEIGHTS: Dict[str, int] = {
    "train": 8,
    "validation": 1,
    "test": 1,
}

INVALID_ANSWER_SENTINELS = {
    "",
    "noanswer",
    "no answer",
    "0 days",
}


# ============================================================
# Standalone temporal_relation -> qtype assignment
# ============================================================

def add_qtype_from_temporal_relation(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Add or overwrite `qtype` for each record using only `temporal_relation`.

    This function is intentionally self-contained:
      - it defines the qtype labels internally;
      - it normalizes and classifies temporal_relation internally;
      - it mutates each record in place and also returns the same record list.

    Classification mapping:
      - timeline / rank_start_time / rank_end_time
            -> timeline_structure
      - union / intersection / sum / average
            -> temporal_operation
      - duration_before / duration_after / duration_during
            -> relative_temporal
      - duration_<N> days before|after
            -> quantitative_temporal
      - interval-topology symbolic relations such as X d Y / X di Y / X o Y / X oi Y
            -> interval_relation
      - pure ordering symbolic relations X < Y / X > Y
            -> temporal_order
      - otherwise
            -> unknown
    """
    qtype_temporal_order = "temporal_order"
    qtype_interval_relation = "interval_relation"
    qtype_relative_temporal = "relative_temporal"
    qtype_quantitative_temporal = "quantitative_temporal"
    qtype_timeline_structure = "timeline_structure"
    qtype_temporal_operation = "temporal_operation"
    qtype_unknown = "unknown"

    timeline_structure_relations = {
        "timeline",
        "rank_start_time",
        "rank_end_time",
    }

    temporal_operation_relations = {
        "union",
        "intersection",
        "sum",
        "average",
    }

    relative_temporal_relations = {
        "duration_before",
        "duration_after",
        "duration_during",
    }

    order_components = {
        "x < y",
        "x > y",
    }

    interval_components = {
        "x d y",
        "x di y",
        "x o y",
        "x oi y",
    }

    def normalize_relation(value: Any) -> str:
        if value is None:
            return ""

        text_value = str(value).strip().lower()
        return re.sub(r"\s+", " ", text_value)

    def split_relation_components(relation: str) -> List[str]:
        if not relation:
            return []

        return [
            re.sub(r"\s+", " ", part.strip())
            for part in relation.split("&")
            if part.strip()
        ]

    def contains_numeric_duration_offset(relation: str) -> bool:
        pattern = r"duration_\d+\s+days\s+(before|after)"
        return re.search(pattern, relation) is not None

    def classify_temporal_relation(relation_raw: Any) -> str:
        relation = normalize_relation(relation_raw)

        if not relation:
            return qtype_unknown

        if relation in timeline_structure_relations:
            return qtype_timeline_structure

        if relation in temporal_operation_relations:
            return qtype_temporal_operation

        if relation in relative_temporal_relations:
            return qtype_relative_temporal

        if contains_numeric_duration_offset(relation):
            return qtype_quantitative_temporal

        components = split_relation_components(relation)

        if any(component in interval_components for component in components):
            return qtype_interval_relation

        if components and all(component in order_components for component in components):
            return qtype_temporal_order

        return qtype_unknown

    for record in records:
        if isinstance(record, dict):
            record["qtype"] = classify_temporal_relation(
                record.get("temporal_relation")
            )

    return records


# ============================================================
# JSON helpers
# ============================================================

def load_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_any(obj: Any, path: Path, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


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
# Stable hashing / allocation
# ============================================================

def stable_json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_record_hash(record: Dict[str, Any]) -> str:
    payload = {
        "id": record.get("id"),
        "question": record.get("question"),
        "question_level": record.get("question_level"),
        "question_type": record.get("question_type"),
        "answer_type": record.get("answer_type"),
        "temporal_relation": record.get("temporal_relation"),
    }
    return hashlib.sha256(stable_json_key(payload).encode("utf-8")).hexdigest()


def allocate_by_largest_remainder(
    n: int,
    weights: Dict[str, int],
    *,
    order: Sequence[str],
) -> Dict[str, int]:
    """
    Allocate `n` items according to integer weights with the largest-remainder method.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    total_weight = sum(weights[k] for k in order)
    if total_weight <= 0:
        raise ValueError("sum(weights) must be positive")

    raw = {
        k: n * weights[k] / total_weight
        for k in order
    }
    floors = {
        k: int(raw[k])
        for k in order
    }

    remaining = n - sum(floors.values())

    # Stable tie break: order listed in `order`.
    order_rank = {name: idx for idx, name in enumerate(order)}
    ranked = sorted(
        order,
        key=lambda k: (raw[k] - floors[k], -order_rank[k]),
        reverse=True,
    )

    for i in range(remaining):
        floors[ranked[i]] += 1

    return floors


# ============================================================
# Existing level adjustment rule
# ============================================================

def should_complex_to_medium(item: Dict[str, Any]) -> bool:
    """
    Rule B:
        complex
        + question_type == timeline_position_retrieval*3
        + answer_type == relation_union_or_intersection
        + temporal_relation == union
        -> medium
    """
    return (
        item.get("question_level") == "complex"
        and item.get("question_type") == "timeline_position_retrieval*3"
        and item.get("answer_type") == "relation_union_or_intersection"
        and item.get("temporal_relation") == "union"
    )


def adjust_question_level(item: Dict[str, Any]) -> Optional[str]:
    if should_complex_to_medium(item):
        return "medium"
    return None


# ============================================================
# Answer normalization / invalid-answer filtering
# ============================================================

def normalize_answer_item(value: Any) -> Any:
    """
    Normalize individual answer items conservatively.

    - String answers are stripped.
    - "equals" becomes "equal".
    - Non-string answers are preserved as-is.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "equals":
            return "equal"
        return stripped

    return value


def answer_to_list(answer: Any) -> List[Any]:
    if isinstance(answer, list):
        return [normalize_answer_item(x) for x in answer]

    return [normalize_answer_item(answer)]


def is_invalid_answer_item(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        normalized = value.strip().casefold()
        return normalized in INVALID_ANSWER_SENTINELS

    return False


def is_invalid_answer_list(answer_list: List[Any]) -> bool:
    """
    Drop:
        []
        [""]
        ["Noanswer"]
        ["No Answer"]
        and lists whose every element is one of those invalid sentinels.
    """
    if not answer_list:
        return True

    return all(is_invalid_answer_item(x) for x in answer_list)


# ============================================================
# Event parsing / sorting
# ============================================================

@dataclass(frozen=True)
class ParsedEvent:
    raw: str
    s: str
    r: str
    o: str
    start_time: str
    end_time: str


def parse_event_str(event_value: Any) -> Optional[ParsedEvent]:
    raw = str(event_value)
    parts = [x.strip() for x in raw.split("|")]
    if len(parts) != 5:
        return None

    return ParsedEvent(
        raw=raw,
        s=parts[0],
        r=parts[1],
        o=parts[2],
        start_time=parts[3],
        end_time=parts[4],
    )


def parse_flexible_date(value: Any) -> Optional[date]:
    """
    Supports:
        YYYY-MM-DD
        YYY-MM-DD
        YYYY-MM
        YYY-MM
        YYYY
        YYY

    Ancient years like 308-01-01 are accepted.
    """
    text = str(value).strip()
    if not text:
        return None

    parts = text.split("-")
    try:
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        if len(parts) == 1:
            return date(int(parts[0]), 1, 1)
    except Exception:
        return None

    return None


def event_sort_key(event_value: Any) -> Tuple[int, Any, str]:
    """
    Sort events by:
        1. start_time ascending
        2. full event string alphabetically ascending

    Invalid dates are placed after valid dates.
    """
    parsed = parse_event_str(event_value)
    if parsed is None:
        return (1, date.max, str(event_value).casefold())

    start_dt = parse_flexible_date(parsed.start_time)
    if start_dt is None:
        return (1, date.max, parsed.raw.casefold())

    return (0, start_dt, parsed.raw.casefold())


def sort_events(events: Any) -> List[Any]:
    if events is None:
        return []

    if isinstance(events, list):
        event_list = list(events)
    else:
        event_list = [events]

    return sorted(event_list, key=event_sort_key)


# ============================================================
# Answer sorting aligned to event entity / relation order
# ============================================================

def build_event_field_order_index(events: List[Any]) -> Dict[str, Tuple[int, int]]:
    """
    Map exact event field value -> first appearance order.

    Field order inside one event:
        subject -> relation -> object
    """
    index: Dict[str, Tuple[int, int]] = {}

    for event_idx, event_value in enumerate(events):
        parsed = parse_event_str(event_value)
        if parsed is None:
            continue

        fields = (parsed.s, parsed.r, parsed.o)
        for field_idx, field_value in enumerate(fields):
            if field_value not in index:
                index[field_value] = (event_idx, field_idx)

    return index


def sort_answers_by_event_field_order(
    answer_list: List[Any],
    events: List[Any],
) -> Tuple[List[Any], bool]:
    """
    Sort answers only when every answer item is a string that can be matched
    exactly to some subject / relation / object field in the sorted events.

    Returns:
        (possibly_sorted_answers, did_sort)
    """
    if len(answer_list) <= 1:
        return answer_list, False

    field_order = build_event_field_order_index(events)

    sortable_rows: List[Tuple[Tuple[int, int], str, Any]] = []
    for answer in answer_list:
        if not isinstance(answer, str):
            return answer_list, False

        if answer not in field_order:
            return answer_list, False

        sortable_rows.append((field_order[answer], answer.casefold(), answer))

    sorted_answers = [
        row[2]
        for row in sorted(sortable_rows, key=lambda row: (row[0], row[1]))
    ]

    did_sort = sorted_answers != answer_list
    return sorted_answers, did_sort


# ============================================================
# Pre-balance cleaning
# ============================================================

def clean_and_normalize_records(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stats = Counter()
    changed_examples: List[Dict[str, Any]] = []
    dropped_examples: List[Dict[str, Any]] = []

    cleaned: List[Dict[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            stats["dropped_non_dict_records"] += 1
            continue

        item = dict(record)

        # -------------------------
        # Existing level adjustment
        # -------------------------
        old_level = item.get("question_level")
        new_level = adjust_question_level(item)
        if new_level is not None and new_level != old_level:
            if "question_level_original" not in item:
                item["question_level_original"] = old_level

            item["question_level"] = new_level
            stats["complex_to_medium_count"] += 1

            if len(changed_examples) < 50:
                changed_examples.append(
                    {
                        "id": item.get("id"),
                        "old_level": old_level,
                        "new_level": new_level,
                        "question_type": item.get("question_type"),
                        "answer_type": item.get("answer_type"),
                        "temporal_relation": item.get("temporal_relation"),
                        "question": item.get("question"),
                    }
                )

        # -------------------------
        # Normalize answer to list
        # -------------------------
        original_answer = item.get("answer")
        answer_list = answer_to_list(original_answer)
        item["answer"] = answer_list

        if not isinstance(original_answer, list):
            stats["scalar_answer_wrapped_as_list"] += 1

        if original_answer == "equals":
            stats["equals_to_equal_scalar"] += 1
        elif isinstance(original_answer, list) and any(x == "equals" for x in original_answer):
            stats["equals_to_equal_inside_list"] += 1

        # -------------------------
        # Drop invalid answers
        # -------------------------
        if is_invalid_answer_list(answer_list):
            stats["dropped_invalid_answer"] += 1
            if len(dropped_examples) < 50:
                dropped_examples.append(
                    {
                        "id": item.get("id"),
                        "reason": "invalid_answer",
                        "answer": answer_list,
                        "question": item.get("question"),
                    }
                )
            continue

        # -------------------------
        # Existing answer size filter
        # -------------------------
        if len(answer_list) > 10:
            stats["dropped_answer_len_gt_10"] += 1
            if len(dropped_examples) < 50:
                dropped_examples.append(
                    {
                        "id": item.get("id"),
                        "reason": "answer_len_gt_10",
                        "answer_len": len(answer_list),
                        "question": item.get("question"),
                    }
                )
            continue

        # -------------------------
        # Remove fields
        # -------------------------
        if "source_kg_id" in item:
            item.pop("source_kg_id", None)
            stats["removed_source_kg_id"] += 1

        if "split" in item:
            item.pop("split", None)
            stats["removed_old_split"] += 1

        # -------------------------
        # Sort events
        # -------------------------
        original_events = item.get("events")
        sorted_events = sort_events(original_events)
        item["events"] = sorted_events

        if isinstance(original_events, list) and sorted_events != original_events:
            stats["events_reordered"] += 1

        # -------------------------
        # Sort answers if exactly alignable to event s/r/o order
        # -------------------------
        sorted_answers, did_sort_answers = sort_answers_by_event_field_order(
            item["answer"],
            sorted_events,
        )
        item["answer"] = sorted_answers

        if did_sort_answers:
            stats["answers_reordered_by_event_field_order"] += 1
        else:
            stats["answers_not_reordered"] += 1

        cleaned.append(item)

    report = {
        "stats": dict(stats),
        "changed_level_examples": changed_examples,
        "dropped_examples": dropped_examples,
    }
    return cleaned, report


# ============================================================
# Level balancing
# ============================================================

def group_by_level(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[str(item.get("question_level"))].append(item)
    return grouped


def question_type_of(item: Dict[str, Any]) -> str:
    return str(item.get("question_type", "__NO_QUESTION_TYPE__"))


def reduce_level_by_question_type(
    records: List[Dict[str, Any]],
    target_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Reduce one level to `target_size`.

    Strategy:
        - bucket by question_type;
        - while oversize:
              remove from the currently largest bucket;
        - ties by question_type lexicographic order;
        - records deleted inside each bucket are deterministic by hash.
    """
    if target_size < 0:
        raise ValueError("target_size must be non-negative")

    if len(records) <= target_size:
        return list(records), [], {}

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in records:
        buckets[question_type_of(item)].append(item)

    # Within each question_type bucket, keep deterministic order.
    # We pop from the end, so larger hashes are deleted first.
    for qtype in buckets:
        buckets[qtype] = sorted(buckets[qtype], key=stable_record_hash)

    removed: List[Dict[str, Any]] = []
    removed_by_qtype = Counter()

    total = sum(len(v) for v in buckets.values())

    while total > target_size:
        max_bucket_size = max(len(v) for v in buckets.values())
        largest_qtypes = sorted(
            qtype
            for qtype, bucket in buckets.items()
            if len(bucket) == max_bucket_size
        )
        chosen_qtype = largest_qtypes[0]

        victim = buckets[chosen_qtype].pop()
        removed.append(victim)
        removed_by_qtype[chosen_qtype] += 1
        total -= 1

    kept: List[Dict[str, Any]] = []
    for qtype in sorted(buckets):
        kept.extend(buckets[qtype])

    # Restore a deterministic global order after balancing.
    kept = sorted(kept, key=stable_record_hash)
    removed = sorted(removed, key=stable_record_hash)

    return kept, removed, dict(removed_by_qtype)


def balance_question_levels(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped = group_by_level(records)

    level_counts_before = {
        level: len(grouped.get(level, []))
        for level in TARGET_LEVELS
    }

    # Only the three requested levels participate in balancing.
    missing_levels = [
        level
        for level, count in level_counts_before.items()
        if count == 0
    ]
    if missing_levels:
        raise ValueError(
            "Cannot balance question_level because at least one target level is empty: "
            + ", ".join(missing_levels)
        )

    target_size = min(level_counts_before.values())

    balanced: List[Dict[str, Any]] = []
    removed_all: List[Dict[str, Any]] = []
    removed_by_level_qtype: Dict[str, Dict[str, int]] = {}

    for level in TARGET_LEVELS:
        kept, removed, removed_by_qtype = reduce_level_by_question_type(
            grouped.get(level, []),
            target_size,
        )
        balanced.extend(kept)
        removed_all.extend(removed)
        removed_by_level_qtype[level] = removed_by_qtype

    # Preserve any unexpected non-target levels, but report them.
    unexpected_levels = sorted(
        level for level in grouped.keys()
        if level not in TARGET_LEVELS
    )
    unexpected_level_records: List[Dict[str, Any]] = []
    for level in unexpected_levels:
        unexpected_level_records.extend(grouped[level])

    if unexpected_level_records:
        balanced.extend(unexpected_level_records)

    balanced = sorted(balanced, key=stable_record_hash)

    level_counts_after = Counter(str(item.get("question_level")) for item in balanced)

    report = {
        "target_size_per_target_level": target_size,
        "level_counts_before": level_counts_before,
        "level_counts_after": dict(level_counts_after),
        "removed_total_for_level_balance": len(removed_all),
        "removed_by_level_and_question_type": removed_by_level_qtype,
        "unexpected_levels_preserved": unexpected_levels,
    }
    return balanced, report


# ============================================================
# Stratified 8:1:1 split
# ============================================================

def stratum_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(item.get("question_level", "__NO_LEVEL__")),
        str(item.get("question_type", "__NO_QUESTION_TYPE__")),
    )


def compute_exact_global_stratified_allocations(
    strata_sizes: Dict[Tuple[str, str], int],
) -> Tuple[Dict[Tuple[str, str], Dict[str, int]], Dict[str, int]]:
    """
    Compute per-stratum train/validation/test allocations that:

    1. preserve stratification as closely as possible;
    2. make the GLOBAL split totals exactly match 8:1:1 under
       largest-remainder rounding.

    Procedure:
        - floor each stratum's fractional quota;
        - compute global deficits against exact global targets;
        - assign each stratum's remaining quota slots to splits with
          positive global deficits, preferring the largest local fractional
          remainder.
    """
    total_records = sum(strata_sizes.values())
    global_targets = allocate_by_largest_remainder(
        total_records,
        SPLIT_WEIGHTS,
        order=SPLIT_NAMES,
    )

    total_weight = sum(SPLIT_WEIGHTS[k] for k in SPLIT_NAMES)

    allocations: Dict[Tuple[str, str], Dict[str, int]] = {}
    fractional_remainders: Dict[Tuple[str, str], Dict[str, float]] = {}
    extra_needed_by_stratum: Dict[Tuple[str, str], int] = {}
    assigned_global = Counter()

    for key in sorted(strata_sizes):
        n = strata_sizes[key]

        raw = {
            split_name: n * SPLIT_WEIGHTS[split_name] / total_weight
            for split_name in SPLIT_NAMES
        }
        floors = {
            split_name: int(raw[split_name])
            for split_name in SPLIT_NAMES
        }

        allocations[key] = dict(floors)
        fractional_remainders[key] = {
            split_name: raw[split_name] - floors[split_name]
            for split_name in SPLIT_NAMES
        }
        extra_needed_by_stratum[key] = n - sum(floors.values())

        for split_name in SPLIT_NAMES:
            assigned_global[split_name] += floors[split_name]

    deficits = {
        split_name: global_targets[split_name] - assigned_global[split_name]
        for split_name in SPLIT_NAMES
    }

    used_extra_splits: Dict[Tuple[str, str], set] = {
        key: set()
        for key in strata_sizes
    }

    remaining_extras = sum(extra_needed_by_stratum.values())

    while remaining_extras > 0:
        candidates: List[Tuple[float, int, str, Tuple[str, str]]] = []

        for key in sorted(strata_sizes):
            if extra_needed_by_stratum[key] <= 0:
                continue

            for split_name in SPLIT_NAMES:
                if deficits[split_name] <= 0:
                    continue

                # A stratum's remaining slots under largest-remainder logic
                # should go to distinct splits.
                if split_name in used_extra_splits[key]:
                    continue

                frac = fractional_remainders[key][split_name]
                candidates.append(
                    (
                        frac,
                        deficits[split_name],
                        split_name,
                        key,
                    )
                )

        if not candidates:
            # This should not occur under ordinary 8:1:1 quotas, but the
            # fallback keeps the implementation robust. It still respects
            # positive global deficits and row capacity.
            fallback_candidates: List[Tuple[int, str, Tuple[str, str]]] = []
            for key in sorted(strata_sizes):
                if extra_needed_by_stratum[key] <= 0:
                    continue
                for split_name in SPLIT_NAMES:
                    if deficits[split_name] > 0:
                        fallback_candidates.append(
                            (deficits[split_name], split_name, key)
                        )

            if not fallback_candidates:
                raise RuntimeError(
                    "Unable to complete stratified split allocation while global deficits remain."
                )

            fallback_candidates.sort(
                key=lambda row: (
                    row[0],
                    -SPLIT_NAMES.index(row[1]),
                    row[2],
                ),
                reverse=True,
            )
            _, chosen_split, chosen_key = fallback_candidates[0]

        else:
            # Prefer:
            #   1. larger local fractional remainder;
            #   2. larger remaining global deficit;
            #   3. stable split priority;
            #   4. stable stratum key.
            candidates.sort(
                key=lambda row: (
                    row[0],
                    row[1],
                    -SPLIT_NAMES.index(row[2]),
                    row[3],
                ),
                reverse=True,
            )
            _, _, chosen_split, chosen_key = candidates[0]

        allocations[chosen_key][chosen_split] += 1
        deficits[chosen_split] -= 1
        extra_needed_by_stratum[chosen_key] -= 1
        used_extra_splits[chosen_key].add(chosen_split)
        remaining_extras -= 1

    if any(value != 0 for value in deficits.values()):
        raise RuntimeError(
            f"Global split allocation deficits not fully resolved: {deficits}"
        )

    return allocations, global_targets


def stratified_split(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in records:
        strata[stratum_key(item)].append(item)

    strata_sizes = {
        key: len(bucket)
        for key, bucket in strata.items()
    }
    allocations, global_target_sizes = compute_exact_global_stratified_allocations(
        strata_sizes
    )

    split_data: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    stratum_allocations: Dict[str, Dict[str, Any]] = {}

    for key in sorted(strata):
        level, qtype = key
        bucket = sorted(strata[key], key=stable_record_hash)
        allocation = allocations[key]

        train_end = allocation["train"]
        validation_end = train_end + allocation["validation"]

        train_items = bucket[:train_end]
        validation_items = bucket[train_end:validation_end]
        test_items = bucket[validation_end:]

        for item in train_items:
            item["split"] = "train"
        for item in validation_items:
            item["split"] = "validation"
        for item in test_items:
            item["split"] = "test"

        split_data["train"].extend(train_items)
        split_data["validation"].extend(validation_items)
        split_data["test"].extend(test_items)

        stratum_allocations[f"{level} || {qtype}"] = {
            "total": len(bucket),
            "train": len(train_items),
            "validation": len(validation_items),
            "test": len(test_items),
        }

    for split_name in SPLIT_NAMES:
        split_data[split_name] = sorted(split_data[split_name], key=stable_record_hash)

    all_records = (
        split_data["train"]
        + split_data["validation"]
        + split_data["test"]
    )
    all_records = sorted(all_records, key=stable_record_hash)

    actual_split_sizes = {
        split_name: len(split_data[split_name])
        for split_name in SPLIT_NAMES
    }

    report = {
        "split_size_targets": global_target_sizes,
        "split_sizes": actual_split_sizes,
        "question_level_distribution_by_split": {
            split_name: dict(Counter(str(item.get("question_level")) for item in split_data[split_name]))
            for split_name in SPLIT_NAMES
        },
        "question_type_distribution_by_split": {
            split_name: dict(Counter(str(item.get("question_type")) for item in split_data[split_name]))
            for split_name in SPLIT_NAMES
        },
        "stratum_allocations": stratum_allocations,
    }

    return all_records, split_data, report


# ============================================================
# Summary statistics
# ============================================================

def level_distribution(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(item.get("question_level")) for item in records))


def question_type_distribution(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(item.get("question_type")) for item in records))


def answer_type_distribution(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(item.get("answer_type")) for item in records))


# ============================================================
# Full pipeline
# ============================================================

def run_final_pipeline(
    input_obj: Any,
) -> Tuple[Any, Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    records, resolved_key = get_records_container(input_obj)

    original_records = [item for item in records if isinstance(item, dict)]

    cleaned_records, clean_report = clean_and_normalize_records(records)
    cleaned_records = add_qtype_from_temporal_relation(cleaned_records)
    balanced_records, balance_report = balance_question_levels(cleaned_records)
    final_all_records, split_data, split_report = stratified_split(balanced_records)

    final_all_obj = replace_records_in_container(
        input_obj,
        final_all_records,
        resolved_key,
    )

    report = {
        "original": {
            "total_records": len(records),
            "dict_records": len(original_records),
            "question_level_distribution": level_distribution(original_records),
            "question_type_distribution": question_type_distribution(original_records),
            "answer_type_distribution": answer_type_distribution(original_records),
        },
        "cleaning": clean_report,
        "after_cleaning_before_level_balance": {
            "total_records": len(cleaned_records),
            "question_level_distribution": level_distribution(cleaned_records),
            "question_type_distribution": question_type_distribution(cleaned_records),
            "answer_type_distribution": answer_type_distribution(cleaned_records),
        },
        "level_balancing": balance_report,
        "final_after_split": {
            "total_records": len(final_all_records),
            "question_level_distribution": level_distribution(final_all_records),
            "question_type_distribution": question_type_distribution(final_all_records),
            "answer_type_distribution": answer_type_distribution(final_all_records),
        },
        "split": split_report,
    }

    return final_all_obj, split_data, report


# ============================================================
# CLI
# ============================================================

def build_output_paths(
    input_path: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    stem = input_path.stem
    return {
        "all": output_dir / f"{stem}.final_all.json",
        "train": output_dir / "train.json",
        "validation": output_dir / "validation.json",
        "test": output_dir / "test.json",
        "report": output_dir / f"{stem}.final_format_report.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final dataset formatting: normalize answers, filter invalid samples, "
            "remove source_kg_id, sort events/answers, balance question_level, "
            "and stratified resplit to train/validation/test=8:1:1."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON dataset path.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to the input file directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    input_obj = load_json_any(input_path)
    final_all_obj, split_data, report = run_final_pipeline(input_obj)

    paths = build_output_paths(input_path, output_dir)

    save_json_any(final_all_obj, paths["all"], indent=2)
    save_json_any(split_data["train"], paths["train"], indent=2)
    save_json_any(split_data["validation"], paths["validation"], indent=2)
    save_json_any(split_data["test"], paths["test"], indent=2)
    save_json_any(report, paths["report"], indent=2)

    final_summary = report["final_after_split"]
    split_summary = report["split"]["split_sizes"]
    balance_summary = report["level_balancing"]

    print("=" * 100)
    print("[Final Dataset Formatting Report]")
    print(f"Input:       {input_path}")
    print(f"Output all:  {paths['all']}")
    print(f"Train:       {paths['train']}")
    print(f"Validation:  {paths['validation']}")
    print(f"Test:        {paths['test']}")
    print(f"Report:      {paths['report']}")
    print("-" * 100)
    print(f"Final total records: {final_summary['total_records']}")
    print(f"Final question_level distribution: {final_summary['question_level_distribution']}")
    print(f"Final split sizes: {split_summary}")
    print(
        "Balanced target per target level: "
        f"{balance_summary['target_size_per_target_level']}"
    )
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
