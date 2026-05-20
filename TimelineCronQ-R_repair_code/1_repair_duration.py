#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repair known question-text generation bugs in the final formatted dataset.

Usage
-----
python repair_duration_question_text_auto_verified.py --input dataset.json

Only `--input` is required.

Outputs
-------
Given:
    /path/to/dataset.json

The script writes:
    /path/to/dataset.question_text_fixed.json
    /path/to/dataset.question_text_fix_report.json

Scope
-----
This script ONLY modifies `question`.
It does NOT modify:
    - answer
    - events
    - question_level
    - question_type
    - answer_type
    - temporal_relation

Medium duration semantics
-------------------------
For medium samples with:
    question_type == timeline_position_retrieval_temporal_constrained_retrieval
    temporal_relation in {duration_after, duration_before}
    answer_type in {object, subject}

the script assumes:
    events[0] = query / answer event
    events[1] = temporal anchor event

Before changing the question, it validates that the offset in the question is
consistent with the events:

    duration_after:
        target_time = anchor_boundary + offset_days
        anchor_boundary:
            - point anchor: anchor.start_time == anchor.end_time
            - interval anchor: anchor.end_time

    duration_before:
        target_time = anchor_boundary - offset_days
        anchor_boundary:
            - point anchor: anchor.start_time == anchor.end_time
            - interval anchor: anchor.start_time

The sample is valid iff:
    query_event.start_time <= target_time <= query_event.end_time

Only offset-valid samples are repaired. Invalid or unparsable samples are left
unchanged and recorded in the audit report.

Implemented repair rules
------------------------

Rule M1A: medium + object + duration_after
    Observed malformed question:
        "{N} days {query_event_phrase}, in which organisation, {query_subject} {query_relation}?"

    Corrected question:
        "{N} days after {anchor_event_phrase}, in which organisation, {query_subject} {query_relation}?"

    This fixes:
        1. the query/anchor phrase reversal;
        2. the missing "after" connector.


Rule M1B: medium + object + duration_before
    Observed malformed question:
        "{N} days {anchor_event_phrase}, in which organisation, {query_subject} {query_relation}?"

    Corrected question:
        "{N} days before {anchor_event_phrase}, in which organisation, {query_subject} {query_relation}?"

    This fixes:
        1. the missing "before" connector.


Rule M2: medium + subject + duration_after / duration_before
    Observed malformed question:
        "Who {query_relation} {query_object} {N} days {anchor_event_phrase}?"

    Corrected question:
        "Who {query_relation} {query_object} {N} days after/before {anchor_event_phrase}?"

    This fixes:
        1. the missing "after" / "before" connector.


Rule C1: complex subject-prefix repair retained from the previous script
    complex
    + question_type == timeline_position_retrieval*2+temporal_constrained_retrieval
    + answer_type == subject

    Observed malformed prefix:
        "Who {query_subject} {query_object}, ..."

    Corrected prefix:
        "Who {query_relation} {query_object}, ..."
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
# Event / date helpers
# ============================================================

@dataclass(frozen=True)
class ParsedEvent:
    s: str
    r: str
    o: str
    start_time: str
    end_time: str


@dataclass(frozen=True)
class OffsetValidation:
    ok: bool
    reason: str
    offset_days: Optional[int] = None
    relation_word: Optional[str] = None
    query_event: Optional[ParsedEvent] = None
    anchor_event: Optional[ParsedEvent] = None
    anchor_boundary_kind: Optional[str] = None
    anchor_boundary_date: Optional[str] = None
    computed_target_date: Optional[str] = None


OFFSET_RE = re.compile(r"(?<!\d)(\d+)\s+days\b", re.IGNORECASE)

OBJECT_DURATION_RE = re.compile(
    r"^(\s*)(\d+)(\s+days\s+)(.+?)(,\s*in which organisation,\s*.+\?\s*)$"
)

SUBJECT_DURATION_RE = re.compile(
    r"^(.+?)\s+(\d+)\s+days\s+(.+?\?\s*)$"
)


def parse_event_str(event_str: Any) -> Optional[ParsedEvent]:
    parts = [x.strip() for x in str(event_str).split("|")]
    if len(parts) != 5:
        return None

    return ParsedEvent(
        s=parts[0],
        r=parts[1],
        o=parts[2],
        start_time=parts[3],
        end_time=parts[4],
    )


def event_phrase(ev: ParsedEvent) -> str:
    return f"{ev.s} {ev.r} {ev.o}"


def collapse_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def parse_flexible_date(text: Any) -> Optional[date]:
    """
    Parse:
        YYYY-MM-DD
        YYY-MM-DD
        YYYY-MM
        YYY-MM
        YYYY
        YYY

    This intentionally supports ancient years such as:
        308-01-01
        440-01-01

    Python's datetime.strptime(..., "%Y-%m-%d") would incorrectly reject
    non-zero-padded years such as "308-01-01".
    """
    raw = str(text).strip()
    if not raw:
        return None

    parts = raw.split("-")

    try:
        if len(parts) == 3:
            year, month, day = (int(parts[0]), int(parts[1]), int(parts[2]))
            return date(year, month, day)

        if len(parts) == 2:
            year, month = (int(parts[0]), int(parts[1]))
            return date(year, month, 1)

        if len(parts) == 1:
            year = int(parts[0])
            return date(year, 1, 1)
    except Exception:
        return None

    return None


def extract_offset_days(question: Any) -> Optional[int]:
    m = OFFSET_RE.search(str(question or ""))
    if not m:
        return None
    return int(m.group(1))


def relation_word_from_temporal_relation(temporal_relation: Any) -> Optional[str]:
    temporal_relation = str(temporal_relation or "")
    if temporal_relation == "duration_after":
        return "after"
    if temporal_relation == "duration_before":
        return "before"
    return None


def validate_medium_duration_offset(sample: Dict[str, Any]) -> OffsetValidation:
    """
    Validate whether the question offset is consistent with events[0]/events[1].

    Assumption:
        events[0] = query / answer event
        events[1] = anchor event
    """
    if sample.get("question_level") != "medium":
        return OffsetValidation(ok=False, reason="not_medium")

    if (
        sample.get("question_type")
        != "timeline_position_retrieval_temporal_constrained_retrieval"
    ):
        return OffsetValidation(ok=False, reason="not_target_question_type")

    if sample.get("answer_type") not in {"object", "subject"}:
        return OffsetValidation(ok=False, reason="not_target_answer_type")

    relation_word = relation_word_from_temporal_relation(sample.get("temporal_relation"))
    if relation_word is None:
        return OffsetValidation(ok=False, reason="not_duration_before_after")

    events = sample.get("events") or []
    if len(events) < 2:
        return OffsetValidation(
            ok=False,
            reason="events_lt_2",
            relation_word=relation_word,
        )

    query_event = parse_event_str(events[0])
    anchor_event = parse_event_str(events[1])

    if query_event is None:
        return OffsetValidation(
            ok=False,
            reason="bad_query_event_format",
            relation_word=relation_word,
        )

    if anchor_event is None:
        return OffsetValidation(
            ok=False,
            reason="bad_anchor_event_format",
            relation_word=relation_word,
            query_event=query_event,
        )

    offset_days = extract_offset_days(sample.get("question", ""))
    if offset_days is None:
        return OffsetValidation(
            ok=False,
            reason="offset_not_found_in_question",
            relation_word=relation_word,
            query_event=query_event,
            anchor_event=anchor_event,
        )

    query_start = parse_flexible_date(query_event.start_time)
    query_end = parse_flexible_date(query_event.end_time)
    anchor_start = parse_flexible_date(anchor_event.start_time)
    anchor_end = parse_flexible_date(anchor_event.end_time)

    if query_start is None or query_end is None:
        return OffsetValidation(
            ok=False,
            reason="bad_query_date",
            offset_days=offset_days,
            relation_word=relation_word,
            query_event=query_event,
            anchor_event=anchor_event,
        )

    if anchor_start is None or anchor_end is None:
        return OffsetValidation(
            ok=False,
            reason="bad_anchor_date",
            offset_days=offset_days,
            relation_word=relation_word,
            query_event=query_event,
            anchor_event=anchor_event,
        )

    if query_start > query_end:
        return OffsetValidation(
            ok=False,
            reason="query_interval_start_after_end",
            offset_days=offset_days,
            relation_word=relation_word,
            query_event=query_event,
            anchor_event=anchor_event,
        )

    if anchor_start > anchor_end:
        return OffsetValidation(
            ok=False,
            reason="anchor_interval_start_after_end",
            offset_days=offset_days,
            relation_word=relation_word,
            query_event=query_event,
            anchor_event=anchor_event,
        )

    if relation_word == "after":
        if anchor_start == anchor_end:
            boundary = anchor_start
            boundary_kind = "anchor_point"
        else:
            boundary = anchor_end
            boundary_kind = "anchor_end"
        target = boundary + timedelta(days=offset_days)

    else:
        if anchor_start == anchor_end:
            boundary = anchor_start
            boundary_kind = "anchor_point"
        else:
            boundary = anchor_start
            boundary_kind = "anchor_start"
        target = boundary - timedelta(days=offset_days)

    ok = query_start <= target <= query_end

    return OffsetValidation(
        ok=ok,
        reason="ok" if ok else "target_not_in_query_interval",
        offset_days=offset_days,
        relation_word=relation_word,
        query_event=query_event,
        anchor_event=anchor_event,
        anchor_boundary_kind=boundary_kind,
        anchor_boundary_date=boundary.isoformat(),
        computed_target_date=target.isoformat(),
    )


def validation_to_report_dict(v: OffsetValidation) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "offset_valid": v.ok,
        "offset_validation_reason": v.reason,
        "offset_days": v.offset_days,
        "relation_word": v.relation_word,
        "anchor_boundary_kind": v.anchor_boundary_kind,
        "anchor_boundary_date": v.anchor_boundary_date,
        "computed_target_date": v.computed_target_date,
    }

    if v.query_event is not None:
        out["query_event_interval"] = [
            v.query_event.start_time,
            v.query_event.end_time,
        ]

    if v.anchor_event is not None:
        out["anchor_event_interval"] = [
            v.anchor_event.start_time,
            v.anchor_event.end_time,
        ]

    return out


# ============================================================
# Medium duration question repair
# ============================================================

def repair_medium_object_duration(
    sample: Dict[str, Any],
    validation: OffsetValidation,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Return:
        (new_question, rule_name, status)
    """
    if not validation.ok:
        return None, None, "offset_invalid"

    if sample.get("answer_type") != "object":
        return None, None, "not_object"

    relation_word = validation.relation_word
    query_event = validation.query_event
    anchor_event = validation.anchor_event

    if relation_word is None or query_event is None or anchor_event is None:
        return None, None, "validation_missing_fields"

    question = str(sample.get("question") or "")
    if re.search(r"\bdays\s+(after|before)\b", question, flags=re.IGNORECASE):
        return None, None, "already_has_connector"

    m = OBJECT_DURATION_RE.match(question)
    if not m:
        return None, None, "object_pattern_not_matched"

    leading_ws = m.group(1)
    offset_digits = m.group(2)
    days_gap = m.group(3)
    original_first_clause = collapse_space(m.group(4))
    tail = m.group(5)

    query_phrase = collapse_space(event_phrase(query_event))
    anchor_phrase = collapse_space(event_phrase(anchor_event))

    if relation_word == "after":
        # Known bug:
        # first clause wrongly uses query event; replace with anchor event.
        if original_first_clause != query_phrase:
            return None, None, "object_after_first_clause_not_query_event"

        new_question = (
            f"{leading_ws}{offset_digits}{days_gap}"
            f"{relation_word} {event_phrase(anchor_event)}{tail}"
        )
        return (
            new_question,
            "M1A_medium_object_duration_after_swap_anchor_and_add_after",
            "repaired",
        )

    # duration_before:
    # first clause already uses anchor event; add "before".
    if original_first_clause != anchor_phrase:
        return None, None, "object_before_first_clause_not_anchor_event"

    new_question = (
        f"{leading_ws}{offset_digits}{days_gap}"
        f"{relation_word} {event_phrase(anchor_event)}{tail}"
    )
    return (
        new_question,
        "M1B_medium_object_duration_before_add_before",
        "repaired",
    )


def repair_medium_subject_duration(
    sample: Dict[str, Any],
    validation: OffsetValidation,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Return:
        (new_question, rule_name, status)
    """
    if not validation.ok:
        return None, None, "offset_invalid"

    if sample.get("answer_type") != "subject":
        return None, None, "not_subject"

    relation_word = validation.relation_word
    anchor_event = validation.anchor_event

    if relation_word is None or anchor_event is None:
        return None, None, "validation_missing_fields"

    question = str(sample.get("question") or "")
    if re.search(r"\bdays\s+(after|before)\b", question, flags=re.IGNORECASE):
        return None, None, "already_has_connector"

    m = SUBJECT_DURATION_RE.match(question)
    if not m:
        return None, None, "subject_pattern_not_matched"

    prefix = m.group(1)
    offset_digits = m.group(2)
    anchor_clause_with_qmark = m.group(3)

    anchor_clause = anchor_clause_with_qmark.strip()
    if anchor_clause.endswith("?"):
        anchor_clause = anchor_clause[:-1].rstrip()

    expected_anchor_phrase = collapse_space(event_phrase(anchor_event))
    if collapse_space(anchor_clause) != expected_anchor_phrase:
        return None, None, "subject_anchor_clause_not_anchor_event"

    new_question = (
        f"{prefix} {offset_digits} days "
        f"{relation_word} {event_phrase(anchor_event)}?"
    )

    rule_name = (
        "M2_medium_subject_duration_after_add_after"
        if relation_word == "after"
        else "M2_medium_subject_duration_before_add_before"
    )
    return new_question, rule_name, "repaired"


# ============================================================
# Existing complex rule retained
# ============================================================

def fix_complex_subject_prefix(
    sample: Dict[str, Any],
) -> Optional[Tuple[str, str]]:
    """
    Return:
        (new_question, rule_name)
    or:
        None
    """
    if not (
        sample.get("question_level") == "complex"
        and sample.get("question_type")
        == "timeline_position_retrieval*2+temporal_constrained_retrieval"
        and sample.get("answer_type") == "subject"
    ):
        return None

    events = sample.get("events") or []
    if not events:
        return None

    query_ev = parse_event_str(events[0])
    if query_ev is None:
        return None

    old_prefix = f"Who {query_ev.s} {query_ev.o},"
    new_prefix = f"Who {query_ev.r} {query_ev.o},"

    question = str(sample.get("question") or "")
    if not question.startswith(old_prefix):
        return None

    new_question = new_prefix + question[len(old_prefix):]
    return new_question, "C1_complex_subject_prefix_relation"


# ============================================================
# Main processing
# ============================================================

def repair_dataset(
    input_obj: Any,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, int]]:
    records, resolved_key = get_records_container(input_obj)

    out_records: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []
    stats = Counter()

    for sample in records:
        if not isinstance(sample, dict):
            out_records.append(sample)
            stats["non_dict_record_skipped"] += 1
            continue

        out = dict(sample)
        old_question = str(out.get("question") or "")
        applied_rules: List[str] = []
        notes: List[str] = []

        # ----------------------------------------------------
        # Medium duration_before / duration_after validation
        # ----------------------------------------------------
        validation = validate_medium_duration_offset(out)

        is_target_medium_duration = (
            out.get("question_level") == "medium"
            and out.get("question_type")
            == "timeline_position_retrieval_temporal_constrained_retrieval"
            and out.get("answer_type") in {"object", "subject"}
            and out.get("temporal_relation")
            in {"duration_after", "duration_before"}
        )

        if is_target_medium_duration:
            stats["medium_duration_target_total"] += 1
            if validation.ok:
                stats["medium_duration_offset_valid"] += 1
            else:
                stats["medium_duration_offset_invalid"] += 1
                stats[f"medium_duration_offset_invalid::{validation.reason}"] += 1

            new_question = None
            rule_name = None
            repair_status = None

            if out.get("answer_type") == "object":
                new_question, rule_name, repair_status = repair_medium_object_duration(
                    out,
                    validation,
                )
            elif out.get("answer_type") == "subject":
                new_question, rule_name, repair_status = repair_medium_subject_duration(
                    out,
                    validation,
                )

            if repair_status is not None:
                stats[f"medium_duration_repair_status::{repair_status}"] += 1

            if new_question is not None and rule_name is not None:
                out["question"] = new_question
                applied_rules.append(rule_name)
                stats[rule_name] += 1

            if validation.ok and new_question is None:
                notes.append(f"offset_valid_but_not_repaired:{repair_status}")

            if not validation.ok:
                notes.append(f"offset_invalid_skip_repair:{validation.reason}")

        # ----------------------------------------------------
        # Existing complex repair rule
        # ----------------------------------------------------
        c1 = fix_complex_subject_prefix(out)
        if c1 is not None:
            new_question, rule_name = c1
            out["question"] = new_question
            applied_rules.append(rule_name)
            stats[rule_name] += 1

        should_report = bool(applied_rules) or is_target_medium_duration

        if should_report:
            entry: Dict[str, Any] = {
                "id": out.get("id"),
                "source_kg_id": out.get("source_kg_id"),
                "question_level": out.get("question_level"),
                "question_type": out.get("question_type"),
                "answer_type": out.get("answer_type"),
                "temporal_relation": out.get("temporal_relation"),
                "old_question": old_question,
                "new_question": out.get("question"),
                "rules": applied_rules,
                "notes": notes,
            }

            if is_target_medium_duration:
                entry.update(validation_to_report_dict(validation))

            report.append(entry)

        if applied_rules:
            stats["changed_total"] += 1

        out_records.append(out)

    repaired_obj = replace_records_in_container(input_obj, out_records, resolved_key)
    return repaired_obj, report, dict(stats)


def build_output_paths(input_path: Path) -> Tuple[Path, Path]:
    stem = input_path.stem
    fixed_path = input_path.with_name(f"{stem}.question_text_fixed.json")
    report_path = input_path.with_name(f"{stem}.question_text_fix_report.json")
    return fixed_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair medium duration_before/after question text after validating "
            "offset consistency against events, while retaining the existing "
            "complex subject-prefix repair."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input final-formatted JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    fixed_path, report_path = build_output_paths(input_path)

    input_obj = load_json_any(input_path)
    repaired_obj, report, stats = repair_dataset(input_obj)

    save_json_any(repaired_obj, fixed_path)
    save_json_any(report, report_path)

    print("=" * 96)
    print("[Question Text Repair Report]")
    print(f"Input:        {input_path}")
    print(f"Fixed JSON:   {fixed_path}")
    print(f"Audit report: {report_path}")
    print("-" * 96)

    if stats:
        for key in sorted(stats):
            print(f"{key}: {stats[key]}")
    else:
        print("No changes applied.")

    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
