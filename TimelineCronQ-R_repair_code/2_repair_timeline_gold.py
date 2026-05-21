#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic gold repair / semantic gold generator for CornQuestionsKG-style test.json.

Core idea:
- Do NOT use LLM extraction.
- Do NOT use source_kg_id.
- Use case["events"] as canonical event facts and relation directions.
- Use case["temporal_relation"] as a hard temporal program.
- Use answer_type to decide projection.
- Optionally load an igraph KG and execute semantic query over the KG.

Outputs per case:
- repaired.source_event_gold: answer directly projected from the dataset's canonical events.
- repaired.semantic_gold: answer from hard KG execution when a graph is provided; otherwise source_event_gold fallback for operator cases.
- repaired.status / repaired.program_type / repaired.notes.

Usage examples:
  python repair_corn_gold.py --input test.json --output test.repaired.json
  python repair_corn_gold.py --input test.json --graph /path/CornQuestionsKG.pkl --output test.repaired.json

Assumptions about graph edge attributes:
- edge["relation"]
- optional edge["timestamp"] in formats supported by normalize_timestamp()
- vertex["name"]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import igraph as ig
from tqdm import tqdm 



from datetime import date
from typing import Union


def parse_date_iso(value: str) -> date:
    """
    Parse ISO-like date string: YYYY-MM-DD or YYY-MM-DD.

    Examples:
        1900-01-01
        308-01-01
    """
    year, month, day = map(int, str(value).strip().split("-"))
    return date(year, month, day)


def compute_duration_days(
    start_time: str,
    end_time: str,
) -> int:
    """
    Unified duration calculation.

    Dataset convention:
        duration_days = end_date - start_date

    Therefore:
        1900-01-01 -> 1900-01-01 = 0 days
        1900-01-01 -> 1900-01-02 = 1 day
        1900-01-01 -> 1901-01-01 = 365 days

    Raises:
        ValueError if end_time < start_time.
    """
    start_dt = parse_date_iso(start_time)
    end_dt = parse_date_iso(end_time)

    if end_dt < start_dt:
        raise ValueError(
            f"Invalid interval: end_time {end_time} < start_time {start_time}"
        )

    return (end_dt - start_dt).days


def format_duration_answer(
    start_time: str,
    end_time: str,
) -> str:
    """
    Convert an interval into benchmark answer format: 'N days'.
    """
    return f"{compute_duration_days(start_time, end_time)} days"


def is_simple_duration_timeline_case(case: Dict[str, Any]) -> bool:
    return (
        case.get("question_level") == "simple"
        and case.get("answer_type") == "duration"
        and case.get("temporal_relation") == "timeline"
    )


def iter_with_progress(iterable, *, total: Optional[int] = None, desc: str = "Processing", enabled: bool = True):
    """tqdm wrapper with a no-dependency fallback."""
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="case")
    return iterable


# ----------------------------- IO -----------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_igraph_pickle(path: str):
    if ig is None:
        raise RuntimeError("python-igraph is not installed; run without --graph or install igraph.")
    # igraph has Graph.Read_Pickle in most versions.
    if hasattr(ig.Graph, "Read_Pickle"):
        return ig.Graph.Read_Pickle(path)
    # Fallback to pickle.
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------- data models -----------------------------

@dataclass(frozen=True)
class Event:
    head: str
    relation: str
    tail: str
    start: str
    end: str
    raw: str


@dataclass(frozen=True)
class Candidate:
    head: str
    relation: str
    tail: str
    start: str
    end: str
    timestamp: str


@dataclass
class GraphExactIndex:
    """Thread-friendly exact index over KG edges.

    It avoids scanning graph.es for every case. Each list contains Candidate
    objects whose timestamps have already been normalized.
    """
    all_edges: List[Candidate]
    by_head: Dict[str, List[Candidate]]
    by_tail: Dict[str, List[Candidate]]
    by_relation: Dict[str, List[Candidate]]


# ----------------------------- date / timestamp parsing -----------------------------

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _last_day_of_month(y: int, m: int) -> int:
    if m == 12:
        return 31
    return (date(y, m + 1, 1) - timedelta(days=1)).day


def _clean_time_text(s: str) -> str:
    s = str(s).strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_single_time_text(raw: str) -> Dict[str, Any]:
    raw = _clean_time_text(raw)

    # YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        y, mo, d = map(int, m.groups())
        dt = date(y, mo, d)
        return {"raw": raw, "start": dt, "end": dt, "granularity": "day", "is_interval": False}

    # YYYY-MM
    m = re.fullmatch(r"(\d{4})-(\d{2})", raw)
    if m:
        y, mo = map(int, m.groups())
        return {
            "raw": raw,
            "start": date(y, mo, 1),
            "end": date(y, mo, _last_day_of_month(y, mo)),
            "granularity": "month",
            "is_interval": False,
        }

    # YYYY
    m = re.fullmatch(r"(\d{4})", raw)
    if m:
        y = int(m.group(1))
        return {"raw": raw, "start": date(y, 1, 1), "end": date(y, 12, 31), "granularity": "year", "is_interval": False}

    # Month YYYY, e.g. Apr 2011
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        mon_s, y_s = m.groups()
        mon = _MONTHS.get(mon_s.lower())
        if mon:
            y = int(y_s)
            return {
                "raw": raw,
                "start": date(y, mon, 1),
                "end": date(y, mon, _last_day_of_month(y, mon)),
                "granularity": "month",
                "is_interval": False,
            }

    # DD Month YYYY, e.g. 27 October 2006
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        d_s, mon_s, y_s = m.groups()
        mon = _MONTHS.get(mon_s.lower())
        if mon:
            dt = date(int(y_s), mon, int(d_s))
            return {"raw": raw, "start": dt, "end": dt, "granularity": "day", "is_interval": False}

    raise ValueError(f"Unsupported single time: {raw}")


def normalize_timestamp(ts: Optional[str]) -> Optional[Dict[str, Any]]:
    if ts is None:
        return None
    raw = _clean_time_text(str(ts))
    if not raw:
        return None

    # Full-date interval without spaces: YYYY-MM-DD-YYYY-MM-DD
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})", raw)
    if m:
        left, right = m.groups()
        try:
            left_norm = _parse_single_time_text(left)
            right_norm = _parse_single_time_text(right)
            if right_norm["end"] < left_norm["start"]:
                return None
            return {
                "raw": ts,
                "start": left_norm["start"],
                "end": right_norm["end"],
                "granularity": None,
                "is_interval": True,
            }
        except Exception:
            return None

    # Year interval: YYYY-YYYY
    m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", raw)
    if m:
        y1, y2 = map(int, m.groups())
        if y2 < y1:
            return None
        return {"raw": ts, "start": date(y1, 1, 1), "end": date(y2, 12, 31), "granularity": None, "is_interval": True}

    # Explicit interval: A to B
    m = re.fullmatch(r"(.+?)\s+to\s+(.+)", raw, flags=re.IGNORECASE)
    if m:
        left, right = m.groups()
        try:
            left_norm = _parse_single_time_text(left)
            right_norm = _parse_single_time_text(right)
            if right_norm["end"] < left_norm["start"]:
                return None
            return {
                "raw": ts,
                "start": left_norm["start"],
                "end": right_norm["end"],
                "granularity": None,
                "is_interval": True,
            }
        except Exception:
            return None

    # Explicit interval: A - B with spaces.
    if " - " in raw:
        left, right = raw.split(" - ", 1)
        try:
            left_norm = _parse_single_time_text(left)
            right_norm = _parse_single_time_text(right)
            if right_norm["end"] < left_norm["start"]:
                return None
            return {
                "raw": ts,
                "start": left_norm["start"],
                "end": right_norm["end"],
                "granularity": None,
                "is_interval": True,
            }
        except Exception:
            pass

    try:
        return _parse_single_time_text(raw)
    except Exception:
        return None


# def parse_date_iso(s: str) -> date:
#     y, m, d = map(int, s.split("-"))
#     return date(y, m, d)


def date_to_iso(d: date) -> str:
    return d.isoformat()


def event_time(e: Event | Candidate) -> Dict[str, Any]:
    return {
        "raw": f"{e.start}-{e.end}",
        "start": parse_date_iso(e.start),
        "end": parse_date_iso(e.end),
        "granularity": None if e.start != e.end else "day",
        "is_interval": e.start != e.end,
    }


def interval_str_from_event(e: Event | Candidate) -> str:
    return f"{e.start} to {e.end}"


def duration_days(e: Event | Candidate) -> int:
    return compute_duration_days(e.start, e.end)


# ----------------------------- event parsing -----------------------------

def parse_event(event_str: str) -> Event:
    parts = event_str.split("|")
    if len(parts) != 5:
        raise ValueError(f"Invalid event string: {event_str}")
    h, r, t, s, e = [x.strip() for x in parts]
    return Event(h, r, t, s, e, event_str)




def canonicalize_answer_item(value: Any) -> str:
    """Canonicalize common answer surface variants before set comparison."""
    s = str(value).strip()
    if not s:
        return ""

    # Full-date intervals: YYYY-MM-DD - YYYY-MM-DD
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})", s)
    if m:
        return f"{m.group(1)} to {m.group(2)}"

    # Full-date intervals already joined without spaces.
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})", s)
    if m:
        return f"{m.group(1)} to {m.group(2)}"

    # Normalize case-insensitive "to" spacing.
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)} to {m.group(2)}"

    return s


def normalize_answer_values(answer: Any) -> List[str]:
    """Normalize raw answer fields into a canonical list of strings.

    Raw CornQuestionsKG exports are not always uniform:
    - repaired/QA-style files often use list[str]
    - earlier raw files may use a single string, sometimes ';'-separated
    - interval separators may be written as " - " rather than " to "

    Canonicalization is used for repair-status comparison only; emitted
    semantic_gold answers already use the target canonical surface form.
    """
    if answer is None:
        return []
    if isinstance(answer, list):
        out: List[str] = []
        for x in answer:
            s = canonicalize_answer_item(x)
            if s:
                out.append(s)
        return out
    if isinstance(answer, tuple):
        return normalize_answer_values(list(answer))
    if isinstance(answer, str):
        raw = answer.strip()
        if not raw:
            return []
        if ";" in raw:
            return [canonicalize_answer_item(x) for x in raw.split(";") if canonicalize_answer_item(x)]
        s = canonicalize_answer_item(raw)
        return [s] if s else []
    s = canonicalize_answer_item(answer)
    return [s] if s else []

def _norm_mention_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _entity_surface_mentioned(entity: str, question: str) -> bool:
    entity_n = _norm_mention_text(entity)
    question_n = _norm_mention_text(question)
    return bool(entity_n) and entity_n in question_n


def prune_unmentioned_events_from_case(case: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Drop seed events whose head and tail are both absent from the question.

    This is intentionally conservative:
    - keep an event when either head OR tail is surface-mentioned;
    - keep malformed event strings so later parsing/debugging can handle them;
    - only remove events that are clearly ungrounded in the question text.
    """
    question = str(case.get("question", "") or "")
    raw_events = list(case.get("events", []) or [])

    kept: List[str] = []
    dropped: List[str] = []

    for raw in raw_events:
        try:
            ev = parse_event(str(raw))
        except Exception:
            kept.append(str(raw))
            continue

        if _entity_surface_mentioned(ev.head, question) or _entity_surface_mentioned(ev.tail, question):
            kept.append(str(raw))
        else:
            dropped.append(str(raw))

    if not dropped:
        return dict(case), []

    out = dict(case)
    out["events"] = kept
    return out, dropped


# ----------------------------- temporal relation parser -----------------------------

ALLEN_MAP = {
    "<": "before",
    ">": "after",
    "=": "equal",
    "d": "during",
    "di": "contains",
    "o": "overlaps",
    "oi": "overlapped_by",
    "s": "starts",
    "si": "started_by",
    "f": "finishes",
    "fi": "finished_by",
    "m": "meets",
}

DURATION_SPECIAL_REL_MAP = {
    "duration_during": "during",
    "duration_finishes": "finishes",
    "duration_starts": "starts",
    "duration_metby": "metby",
}

STANDALONE_OPERATORS = {
    "average",
    "duration",
    "duration_compare",
    "intersection",
    "rank_end_time",
    "rank_start_time",
    "sum",
    "timeline",
    "union",
}

SINGLE_OFFSET_RELATIONS = {
    "duration_after": "after",
    "duration_before": "before",
}


def parse_temporal_relation_unit(unit: str, anchor_index: int) -> Dict[str, Any]:
    unit = unit.strip()

    m = re.fullmatch(r"duration_(\d+)\s+days\s+(after|before)", unit)
    if m:
        return {
            "kind": "constraint",
            "relation": m.group(2),
            "anchor_index": anchor_index,
            "offset_days": int(m.group(1)),
            "raw": unit,
        }

    if unit in DURATION_SPECIAL_REL_MAP:
        return {
            "kind": "constraint",
            "relation": DURATION_SPECIAL_REL_MAP[unit],
            "anchor_index": anchor_index,
            "offset_days": None,
            "raw": unit,
        }

    m = re.fullmatch(r"X\s+([a-zA-Z<>=]+)\s+Y", unit)
    if m:
        sym = m.group(1)
        if sym not in ALLEN_MAP:
            return {"kind": "unsupported", "raw": unit, "reason": f"unknown_allen_symbol:{sym}"}
        return {
            "kind": "constraint",
            "relation": ALLEN_MAP[sym],
            "anchor_index": anchor_index,
            "offset_days": None,
            "raw": unit,
        }

    if unit in STANDALONE_OPERATORS:
        return {"kind": "operator", "operator": unit, "raw": unit}

    if unit in SINGLE_OFFSET_RELATIONS:
        return {
            "kind": "constraint",
            "relation": SINGLE_OFFSET_RELATIONS[unit],
            "anchor_index": 0,
            "offset_days": None,
            "needs_question_offset": True,
            "raw": unit,
        }

    return {"kind": "unsupported", "raw": unit, "reason": "no_parser_matched"}


def parse_temporal_relation(rel: Optional[str]) -> Dict[str, Any]:
    rel = (rel or "").strip()
    if not rel:
        return {"operator": None, "constraints": [], "unsupported": []}

    if rel in STANDALONE_OPERATORS:
        return {"operator": rel, "constraints": [], "unsupported": []}

    if rel in SINGLE_OFFSET_RELATIONS:
        return {
            "operator": None,
            "constraints": [parse_temporal_relation_unit(rel, 0)],
            "unsupported": [],
        }

    parts = [p.strip() for p in rel.split("&") if p.strip()]
    constraints: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, Any]] = []
    operator = None

    for i, part in enumerate(parts):
        parsed = parse_temporal_relation_unit(part, i)
        if parsed["kind"] == "constraint":
            constraints.append(parsed)
        elif parsed["kind"] == "operator":
            operator = parsed["operator"]
        else:
            unsupported.append(parsed)

    return {"operator": operator, "constraints": constraints, "unsupported": unsupported}


# ----------------------------- temporal matching -----------------------------

def match_allen_relation(x_time: Dict[str, Any], y_time: Dict[str, Any], relation: str) -> bool:
    xs, xe = x_time["start"], x_time["end"]
    ys, ye = y_time["start"], y_time["end"]

    if relation == "before":
        return xe < ys
    if relation == "after":
        return xs > ye
    if relation == "equal":
        return xs == ys and xe == ye
    if relation == "during":
        return xs >= ys and xe <= ye
    if relation == "contains":
        return xs <= ys and xe >= ye
    if relation == "overlaps":
        return xs < ys <= xe < ye
    if relation == "overlapped_by":
        return ys < xs <= ye < xe
    if relation == "starts":
        return xs == ys and xe <= ye
    if relation == "started_by":
        return xs == ys and xe >= ye
    if relation == "finishes":
        return xs >= ys and xe == ye
    if relation == "finished_by":
        return xs <= ys and xe == ye
    if relation == "meets":
        return xe == ys
    if relation == "metby":
        return xs == ye
    raise ValueError(f"Unsupported relation: {relation}")



START_CUE_RE = re.compile(
    r"\b(begin|began|begun|start|started|starts|starting|commenced|commence)\b",
    re.IGNORECASE,
)

END_CUE_RE = re.compile(
    r"\b(end|ended|ending|finish|finished|finishes|stop|stopped|stops|ceased|cease|left|leaving)\b",
    re.IGNORECASE,
)


def detect_offset_boundary_cue(question: str) -> Optional[str]:
    q = question or ""
    has_start = bool(START_CUE_RE.search(q))
    has_end = bool(END_CUE_RE.search(q))
    if has_start and not has_end:
        return "start"
    if has_end and not has_start:
        return "end"
    return None


def resolve_offset_anchor_boundary(question: str, relation: str, y_time: Dict[str, Any]) -> str:
    cue = detect_offset_boundary_cue(question)
    if cue in {"start", "end"}:
        return cue

    if y_time["start"] == y_time["end"]:
        return "point"

    # Natural-language default:
    #   N days after interval anchor  -> anchor end
    #   N days before interval anchor -> anchor start
    return "end" if relation == "after" else "start"


def match_offset_relation(
    x_time: Dict[str, Any],
    y_time: Dict[str, Any],
    relation: str,
    offset_days: int,
    question: str = "",
) -> bool:
    boundary = resolve_offset_anchor_boundary(question, relation, y_time)
    if boundary in {"start", "point"}:
        base = y_time["start"]
    elif boundary == "end":
        base = y_time["end"]
    else:
        raise ValueError(f"Unsupported offset boundary: {boundary}")

    if relation == "after":
        target = base + timedelta(days=offset_days)
    elif relation == "before":
        target = base - timedelta(days=offset_days)
    else:
        raise ValueError(f"Offset relation must be after/before, got {relation}")

    xs, xe = x_time["start"], x_time["end"]
    return xs <= target <= xe


def match_constraint(
    x: Candidate | Event,
    y: Event,
    constraint: Dict[str, Any],
    question: str = "",
) -> bool:
    xt = event_time(x)
    yt = event_time(y)
    offset = constraint.get("offset_days")
    relation = constraint["relation"]
    if offset is not None:
        return match_offset_relation(xt, yt, relation, int(offset), question=question)
    return match_allen_relation(xt, yt, relation)

# ----------------------------- text linking utilities -----------------------------

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def event_score_in_text(text: str, e: Event) -> int:
    t = norm_text(text)
    score = 0
    if norm_text(e.head) and norm_text(e.head) in t:
        score += 2
    if norm_text(e.tail) and norm_text(e.tail) in t:
        score += 2
    # relation signal is deliberately weak; canonical direction comes from events.
    rel = norm_text(e.relation)
    if rel and rel in t:
        score += 1
    return score

def _event_first_mention_pos(text: str, e: Event) -> int:
    """Return earliest mention position of event head/tail in question text.

    This is intentionally entity-based rather than syntax-based: event direction
    remains canonical from `events`, while question order is used only for
    option numbering in duration_compare / rank-like questions.
    """
    t = norm_text(text)
    positions: List[int] = []
    for mention in (e.head, e.tail):
        m = norm_text(mention)
        if not m:
            continue
        pos = t.find(m)
        if pos >= 0:
            positions.append(pos)
    return min(positions) if positions else 10**12


def event_indices_in_question_order(question: str, events: List[Event]) -> List[int]:
    """Order event indices by their first entity mention in question.

    For `Which one lasted ... among A, B, C`, the dataset's `events` order may
    differ from the option order in the question. This function gives the option
    order from the surface question. Unmatched events are placed after matched
    events in their original order.
    """
    scored = [(_event_first_mention_pos(question, e), i) for i, e in enumerate(events)]
    scored.sort(key=lambda x: (x[0], x[1]))
    return [i for _, i in scored]


_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


def _parse_ordinal_token(tok: str) -> Optional[int]:
    tok = tok.strip().lower()
    if tok in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[tok]
    m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", tok)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"\d+", tok)
    if m:
        return int(tok)
    return None


def parse_ranked_duration_compare(question: str) -> Optional[Tuple[int, str]]:
    """Parse templates like: Which one is the 2nd longest/shortest among ..."""
    q = question.lower()
    m = re.search(
        r"which one\s+is\s+the\s+([a-z]+|\d+(?:st|nd|rd|th)?)\s+(longest|shortest)\s+among",
        q,
    )
    if not m:
        return None
    n = _parse_ordinal_token(m.group(1))
    if n is None:
        return None
    return n, m.group(2)


def is_which_one_duration_compare(question: str) -> bool:
    """Subtype A: Which one lasted the longest/shortest among ..."""
    q = question.lower()
    return (
        "which one" in q
        and (
            "lasted the longest" in q
            or "lasted the shortest" in q
            or "lasted longest" in q
            or "lasted shortest" in q
        )
    )


def is_binary_duration_compare(question: str) -> bool:
    """Subtype B: Is the duration of A longer/shorter/equal to B?"""
    q = question.lower()
    return (
        "is the duration" in q
        and (
            "longer than" in q
            or "shorter than" in q
            or "equal to" in q
        )
    )


def duration_compare_subtype(question: str) -> str:
    if parse_ranked_duration_compare(question) is not None:
        return "which_one_ranked_longest_shortest"
    if is_which_one_duration_compare(question):
        return "which_one_longest_shortest"
    if is_binary_duration_compare(question):
        return "binary_longer_shorter_equal"
    return "unsupported_duration_compare"


def extract_offset_from_question(question: str) -> Optional[Tuple[int, str]]:
    m = re.search(r"(\d+)\s+days\s+(before|after)", question, flags=re.I)
    if not m:
        return None
    return int(m.group(1)), m.group(2).lower()


def split_single_offset_question(question: str) -> Tuple[str, str]:
    parts = question.split(",", 1)
    if len(parts) == 1:
        return question, question
    return parts[0].strip(), parts[1].strip()


def temporal_anchor_spans(question: str) -> List[str]:
    # Conservative marker slicing for before/after/during/same-time structures.
    marker_re = re.compile(
        r"\b(?:\d+\s+days\s+(?:before|after)|before|after|during|while|at the same time that|between)\b",
        re.I,
    )
    matches = list(marker_re.finditer(question))
    spans: List[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        span = question[start:end].strip()
        # Cut only very hard separators. Do not cut on every comma too aggressively.
        span = re.split(r";", span, maxsplit=1)[0].strip()
        spans.append(span)
    return spans


def link_anchors_from_question(question: str, events: List[Event], expected_count: int) -> Tuple[List[int], Dict[str, Any]]:
    spans = temporal_anchor_spans(question)
    used: set[int] = set()
    anchors: List[int] = []
    debug: List[Dict[str, Any]] = []

    for span in spans:
        scored = []
        for i, e in enumerate(events):
            if i in used:
                continue
            scored.append((event_score_in_text(span, e), i, e.raw))
        scored.sort(key=lambda x: x[0], reverse=True)
        debug.append({"span": span, "top_scores": scored[:5]})
        if scored and scored[0][0] > 0:
            anchors.append(scored[0][1])
            used.add(scored[0][1])
        if len(anchors) >= expected_count:
            break

    return anchors, {"spans": spans, "link_debug": debug}


def infer_query_index(events: List[Event], anchor_indices: Sequence[int]) -> Optional[int]:
    used = set(anchor_indices)
    remain = [i for i in range(len(events)) if i not in used]
    return remain[0] if len(remain) == 1 else None


def find_rank_target_index(question: str, events: List[Event]) -> Optional[int]:
    q = question.lower()
    m = re.search(r"event (?:in which|where) (.+?) among", q, flags=re.I)
    span = m.group(1) if m else q
    scored = []
    for i, e in enumerate(events):
        scored.append((event_score_in_text(span, e), i))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] == 0:
        return None
    return scored[0][1]


# ----------------------------- query construction and graph execution -----------------------------

def build_query_template(source_event: Event, answer_type: str) -> Tuple[Dict[str, str], str]:
    if answer_type == "subject":
        return {"head": "?", "relation": source_event.relation, "tail": source_event.tail}, "head"
    if answer_type == "object":
        return {"head": source_event.head, "relation": source_event.relation, "tail": "?"}, "tail"
    # time-like: exact event.
    return {"head": source_event.head, "relation": source_event.relation, "tail": source_event.tail}, "time"



# Conservative relation-direction fallbacks.
# Extend only after verifying schema-level direction variation in the KG.
INVERSE_RELATIONS = {
    "part of": "has part",
    "has part": "part of",
    "spouse": "spouse of",
    "spouse of": "spouse",
    "owned by": "owner of",
    "owner of": "owned by",
}

# Relations that may be stored with the same label but reversed argument order.
# Confirmed example in this KG family:
#   Person | winner | Award
#   Award  | winner | Person
SAME_RELATION_REVERSE = {
    "winner",
}


def inverse_query_pattern(query: Dict[str, str], inverse_relation: str) -> Dict[str, str]:
    return {
        "head": query["tail"],
        "relation": inverse_relation,
        "tail": query["head"],
    }


def _edge_attr(edge: Any, name: str, default=None):
    try:
        if name in edge.attribute_names():
            return edge[name]
    except Exception:
        pass
    return default



def normalize_edge_time(edge: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """Read temporal information from the KG edge.

    Preferred schema:
        edge["timestamp"]
    Fallback schema:
        edge["start_time"], edge["end_time"]

    Your current CornQuestionsKG.pkl exposes both, so the preferred path will
    normally be used; the fallback keeps the script robust to future exports.
    """
    ts = _edge_attr(edge, "timestamp")
    norm = normalize_timestamp(ts)
    if norm is not None:
        return norm, str(ts)

    start = _edge_attr(edge, "start_time")
    end = _edge_attr(edge, "end_time")
    if start is not None and end is not None:
        fallback_raw = f"{start} to {end}"
        norm = normalize_timestamp(fallback_raw)
        if norm is not None:
            return norm, fallback_raw

    return None, str(ts) if ts is not None else ""


def exact_scan_graph(graph: Any, query: Dict[str, str]) -> List[Candidate]:
    if graph is None:
        return []
    out: List[Candidate] = []

    # Generic igraph scan. This is not fastest, but it is robust and enough for gold repair.
    for edge in graph.es:
        try:
            h = graph.vs[edge.source]["name"]
            t = graph.vs[edge.target]["name"]
            r = _edge_attr(edge, "relation")
            norm, raw_ts = normalize_edge_time(edge)
        except Exception:
            continue

        if query["head"] != "?" and h != query["head"]:
            continue
        if query["relation"] != "?" and r != query["relation"]:
            continue
        if query["tail"] != "?" and t != query["tail"]:
            continue

        if norm is None:
            continue

        out.append(
            Candidate(
                head=h,
                relation=r,
                tail=t,
                start=date_to_iso(norm["start"]),
                end=date_to_iso(norm["end"]),
                timestamp=raw_ts,
            )
        )
    return out


def build_graph_exact_index(graph: Any, show_progress: bool = True) -> GraphExactIndex:
    """Build a read-only exact index for fast semantic gold retrieval.

    This is usually much faster than scanning graph.es for every query.
    It normalizes timestamps once and keeps only edges with parseable time.
    """
    all_edges: List[Candidate] = []
    by_head: Dict[str, List[Candidate]] = defaultdict(list)
    by_tail: Dict[str, List[Candidate]] = defaultdict(list)
    by_relation: Dict[str, List[Candidate]] = defaultdict(list)

    iterator = iter_with_progress(
        graph.es,
        total=len(graph.es),
        desc="Indexing KG edges",
        enabled=show_progress,
    )

    for edge in iterator:
        try:
            h = graph.vs[edge.source]["name"]
            t = graph.vs[edge.target]["name"]
            r = _edge_attr(edge, "relation")
            norm, raw_ts = normalize_edge_time(edge)
        except Exception:
            continue

        if r is None:
            continue

        if norm is None:
            continue

        cand = Candidate(
            head=h,
            relation=r,
            tail=t,
            start=date_to_iso(norm["start"]),
            end=date_to_iso(norm["end"]),
            timestamp=raw_ts,
        )
        all_edges.append(cand)
        by_head[h].append(cand)
        by_tail[t].append(cand)
        by_relation[r].append(cand)

    return GraphExactIndex(
        all_edges=all_edges,
        by_head=dict(by_head),
        by_tail=dict(by_tail),
        by_relation=dict(by_relation),
    )


def exact_scan_index(index: GraphExactIndex, query: Dict[str, str]) -> List[Candidate]:
    """Fast exact query over the prebuilt edge index.

    Query supports '?' wildcard in head/relation/tail.
    The smallest constrained posting list is chosen as the candidate pool.
    """
    pools: List[List[Candidate]] = []

    if query["head"] != "?":
        pools.append(index.by_head.get(query["head"], []))
    if query["relation"] != "?":
        pools.append(index.by_relation.get(query["relation"], []))
    if query["tail"] != "?":
        pools.append(index.by_tail.get(query["tail"], []))

    if pools:
        pool = min(pools, key=len)
    else:
        pool = index.all_edges

    out: List[Candidate] = []
    qh, qr, qt = query["head"], query["relation"], query["tail"]
    for cand in pool:
        if qh != "?" and cand.head != qh:
            continue
        if qr != "?" and cand.relation != qr:
            continue
        if qt != "?" and cand.tail != qt:
            continue
        out.append(cand)
    return out




def dedup_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    seen: set[Tuple[str, str, str, str, str]] = set()
    out: List[Candidate] = []
    for c in sorted(candidates, key=_candidate_sort_key):
        key = (c.head, c.relation, c.tail, c.start, c.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def exact_scan_index_with_fallback(index: GraphExactIndex, query: Dict[str, str]) -> Tuple[List[Candidate], List[str]]:
    """Exact KG query with conservative direction fallbacks."""
    notes: List[str] = []

    exact = exact_scan_index(index, query)
    if exact:
        return dedup_candidates(exact), notes

    relation = query.get("relation")
    if relation in SAME_RELATION_REVERSE:
        reverse_query = inverse_query_pattern(query, relation)
        reversed_same = exact_scan_index(index, reverse_query)
        if reversed_same:
            notes.append(
                f"matched_reversed_same_relation_pattern:"
                f"{query['head']}|{relation}|{query['tail']}"
                f"->{reverse_query['head']}|{reverse_query['relation']}|{reverse_query['tail']}"
            )
            return dedup_candidates(reversed_same), notes

    inverse_relation = INVERSE_RELATIONS.get(str(relation))
    if inverse_relation:
        inverse_query = inverse_query_pattern(query, inverse_relation)
        inverse = exact_scan_index(index, inverse_query)
        if inverse:
            notes.append(
                f"matched_inverse_relation_pattern:"
                f"{query['head']}|{relation}|{query['tail']}"
                f"->{inverse_query['head']}|{inverse_query['relation']}|{inverse_query['tail']}"
            )
            return dedup_candidates(inverse), notes

    return [], notes


def exact_scan_graph_with_fallback(graph: Any, query: Dict[str, str]) -> Tuple[List[Candidate], List[str]]:
    """Graph-scan KG query with conservative direction fallbacks."""
    notes: List[str] = []

    exact = exact_scan_graph(graph, query)
    if exact:
        return dedup_candidates(exact), notes

    relation = query.get("relation")
    if relation in SAME_RELATION_REVERSE:
        reverse_query = inverse_query_pattern(query, relation)
        reversed_same = exact_scan_graph(graph, reverse_query)
        if reversed_same:
            notes.append(
                f"matched_reversed_same_relation_pattern:"
                f"{query['head']}|{relation}|{query['tail']}"
                f"->{reverse_query['head']}|{reverse_query['relation']}|{reverse_query['tail']}"
            )
            return dedup_candidates(reversed_same), notes

    inverse_relation = INVERSE_RELATIONS.get(str(relation))
    if inverse_relation:
        inverse_query = inverse_query_pattern(query, inverse_relation)
        inverse = exact_scan_graph(graph, inverse_query)
        if inverse:
            notes.append(
                f"matched_inverse_relation_pattern:"
                f"{query['head']}|{relation}|{query['tail']}"
                f"->{inverse_query['head']}|{inverse_query['relation']}|{inverse_query['tail']}"
            )
            return dedup_candidates(inverse), notes

    return [], notes


def scan_semantic_candidates(
    *,
    graph: Any,
    graph_index: Optional[GraphExactIndex],
    query: Dict[str, str],
) -> Tuple[List[Candidate], str, List[str]]:
    if graph_index is not None:
        candidates, notes = exact_scan_index_with_fallback(graph_index, query)
        return candidates, "semantic_graph_exact_indexed", notes
    if graph is not None:
        candidates, notes = exact_scan_graph_with_fallback(graph, query)
        return candidates, "semantic_graph_exact", notes
    return [], "source_event_no_graph", ["graph_not_provided"]


def project_candidate(c: Candidate, projection: str, answer_type: str) -> List[str]:
    if projection == "head":
        return [c.head]
    if projection == "tail":
        return [c.tail]
    if projection == "time":
        if answer_type in {"timestamp", "timestamp_start"}:
            return [c.start]
        if answer_type == "timestamp_end":
            return [c.end]

        if answer_type in {"timestamp_range"}:
        # if answer_type in {"timestamp_range", "duration"}:
            return [f"{c.start} to {c.end}"]
        if answer_type == "duration":
            return [format_duration_answer(c.start, c.end)]
        return [c.start]
    return []


def dedup_keep_order(xs: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _candidate_sort_key(c: Candidate) -> Tuple[str, str, str, str, str]:
    return (c.start, c.end, c.head, c.relation, c.tail)


def _event_sort_key(e: Event) -> Tuple[str, str, str, str, str]:
    return (e.start, e.end, e.head, e.relation, e.tail)


def candidate_to_event_string(c: Candidate) -> str:
    """Serialize a KG candidate back to the dataset event format."""
    return f"{c.head}|{c.relation}|{c.tail}|{c.start}|{c.end}"


def semantic_gold_events_from_candidates(
    candidates: Sequence[Candidate],
    projection: str,
    answer_type: str,
    semantic_gold: Sequence[str],
) -> List[str]:
    """Return one sorted event string for each semantic answer.

    Candidates are sorted by start_time. If multiple KG events project to the
    same answer string, keep the earliest event only.
    """
    wanted = set(semantic_gold or [])
    if not wanted:
        return []

    seen_answers: set[str] = set()
    out: List[str] = []
    for cand in sorted(candidates, key=_candidate_sort_key):
        projected = project_candidate(cand, projection, answer_type)
        hit = False
        for ans in projected:
            if ans in wanted and ans not in seen_answers:
                seen_answers.add(ans)
                hit = True
        if hit:
            out.append(candidate_to_event_string(cand))

        if seen_answers >= wanted:
            break

    return out


def project_candidates_sorted_by_start(
    candidates: Sequence[Candidate],
    projection: str,
    answer_type: str,
) -> List[str]:
    """Project candidates after sorting by start_time ascending."""
    answers: List[str] = []
    for cand in sorted(candidates, key=_candidate_sort_key):
        answers.extend(project_candidate(cand, projection, answer_type))
    return dedup_keep_order(answers)



def unique_seed_events_by_sro(events: Sequence[Event]) -> List[Event]:
    seen: set[Tuple[str, str, str]] = set()
    out: List[Event] = []
    for e in events:
        key = (e.head, e.relation, e.tail)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def candidate_event_strings(candidates: Sequence[Candidate]) -> List[str]:
    return [candidate_to_event_string(c) for c in dedup_candidates(candidates)]


def query_exact_sro_group(
    seed: Event,
    *,
    graph: Any,
    graph_index: Optional[GraphExactIndex],
) -> Tuple[List[Candidate], str, List[str]]:
    query = {"head": seed.head, "relation": seed.relation, "tail": seed.tail}
    return scan_semantic_candidates(graph=graph, graph_index=graph_index, query=query)


def query_timeline_group(
    seed: Event,
    answer_type: str,
    *,
    graph: Any,
    graph_index: Optional[GraphExactIndex],
) -> Tuple[List[Candidate], str, List[str], str]:
    """KG completion for temporal_relation == timeline.

    For entity-answer tasks, preserve the seed interval. This matches the
    original simple temporal-constrained retrieval semantics: complete all
    answers sharing the same relation pattern and exact start/end interval.

    For time-answer tasks, retrieve all timestamps for the exact SRO.
    """
    if answer_type == "subject":
        query = {"head": "?", "relation": seed.relation, "tail": seed.tail}
        projection = "head"
        candidates, mode, notes = scan_semantic_candidates(graph=graph, graph_index=graph_index, query=query)
        candidates = [c for c in candidates if c.start == seed.start and c.end == seed.end]
        return dedup_candidates(candidates), mode, notes, projection

    if answer_type == "object":
        query = {"head": seed.head, "relation": seed.relation, "tail": "?"}
        projection = "tail"
        candidates, mode, notes = scan_semantic_candidates(graph=graph, graph_index=graph_index, query=query)
        candidates = [c for c in candidates if c.start == seed.start and c.end == seed.end]
        return dedup_candidates(candidates), mode, notes, projection

    query = {"head": seed.head, "relation": seed.relation, "tail": seed.tail}
    candidates, mode, notes = scan_semantic_candidates(graph=graph, graph_index=graph_index, query=query)
    return dedup_candidates(candidates), mode, notes, "time"


def merge_intervals_from_candidates(candidates: Sequence[Candidate]) -> List[Tuple[str, str]]:
    if not candidates:
        return []
    intervals = sorted([(c.start, c.end) for c in dedup_candidates(candidates)], key=lambda x: (x[0], x[1]))
    merged: List[Tuple[str, str]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def duration_days_from_iso_interval(start: str, end: str) -> int:
    """Unified interval-duration helper used by KG-completed duration operators.

    Dataset convention:
        duration_days = end_date - start_date
        point interval such as 1900-01-01 to 1900-01-01 = 0 days
    """
    return compute_duration_days(start, end)


def group_duration_days_from_candidates(candidates: Sequence[Candidate]) -> int:
    merged = merge_intervals_from_candidates(candidates)
    return sum(duration_days_from_iso_interval(s, e) for s, e in merged)


def average_duration_value(values: Sequence[int], mode: str = "floor") -> str:
    vals = list(values)
    if not vals:
        raise ValueError("empty duration values")
    total = sum(vals)
    n = len(vals)
    if mode == "floor":
        return f"{total // n} days"
    if mode == "round":
        return f"{round(total / n)} days"
    if mode == "float":
        return f"{total / n:.2f} days"
    raise ValueError(f"Unsupported average mode: {mode}")


def execute_operator_with_graph(
    case: Dict[str, Any],
    events: List[Event],
    operator: str,
    *,
    graph: Any,
    graph_index: Optional[GraphExactIndex],
    duration_tie_policy: str,
    duration_average_mode: str = "floor",
) -> Tuple[List[str], Dict[str, Any]]:
    """Execute operator programs with KG completion where the semantics are stable.

    KG-completed operators:
      - timeline
      - union
      - intersection
      - sum
      - average

    Source-event deterministic operators retained as-is:
      - duration
      - duration_compare
      - rank_start_time
      - rank_end_time

    The retained source-event operators are intentionally not KG-expanded:
    their answer semantics are option- or event-order-sensitive, and naïvely
    expanding all KG timestamps can change the intended problem definition.
    """
    answer_type = str(case.get("answer_type", "") or "")
    debug: Dict[str, Any] = {
        "operator": operator,
        "graph_completion_used": graph is not None or graph_index is not None,
        "semantic_gold_events": [],
        "query_notes": [],
        "missing_groups": [],
    }

    # No graph: preserve deterministic source-event behavior.
    if graph is None and graph_index is None:
        ans, source_dbg = answer_operator_from_events(
            case,
            events,
            operator,
            duration_tie_policy=duration_tie_policy,
        )
        debug["mode"] = "source_event_no_graph"
        debug["operator_debug"] = source_dbg
        debug["semantic_gold_events"] = [e.raw for e in sorted(events, key=_event_sort_key)]
        return ans, debug

    # timeline completion
    if operator == "timeline":
        all_candidates: List[Candidate] = []
        answers: List[str] = []
        scan_modes: List[str] = []
        for seed in unique_seed_events_by_sro(events):
            candidates, mode, notes, projection = query_timeline_group(
                seed,
                answer_type,
                graph=graph,
                graph_index=graph_index,
            )
            scan_modes.append(mode)
            debug["query_notes"].extend(notes)
            if not candidates:
                debug["missing_groups"].append(seed.raw)
                continue
            all_candidates.extend(candidates)
            answers.extend(project_candidates_sorted_by_start(candidates, projection, answer_type))

        debug["mode"] = ",".join(sorted(set(scan_modes))) if scan_modes else "semantic_graph_exact"
        debug["candidate_count"] = len(dedup_candidates(all_candidates))
        debug["semantic_gold_events"] = candidate_event_strings(all_candidates)

        if debug["missing_groups"]:
            debug["warning"] = "operator_timeline_missing_kg_groups"
            return [], debug

        return dedup_keep_order(answers), debug

    # Union/intersection completion over all KG facts for each seed SRO.
    if operator in {"union", "intersection", "sum", "average"}:
        groups: List[List[Candidate]] = []
        scan_modes: List[str] = []
        all_candidates: List[Candidate] = []
        for seed in unique_seed_events_by_sro(events):
            candidates, mode, notes = query_exact_sro_group(
                seed,
                graph=graph,
                graph_index=graph_index,
            )
            scan_modes.append(mode)
            debug["query_notes"].extend(notes)
            if not candidates:
                debug["missing_groups"].append(seed.raw)
                continue
            candidates = dedup_candidates(candidates)
            groups.append(candidates)
            all_candidates.extend(candidates)

        debug["mode"] = ",".join(sorted(set(scan_modes))) if scan_modes else "semantic_graph_exact"
        debug["candidate_count"] = len(dedup_candidates(all_candidates))
        debug["group_count"] = len(groups)
        debug["semantic_gold_events"] = candidate_event_strings(all_candidates)

        if debug["missing_groups"]:
            debug["warning"] = "operator_missing_kg_groups"
            return [], debug

        if operator == "union":
            expanded_as_event_like: List[Candidate] = []
            for group in groups:
                expanded_as_event_like.extend(group)
            return union_intervals(expanded_as_event_like), debug

        if operator == "intersection":
            if not groups:
                return [], debug
            # Intersect the per-SRO temporal coverage groups.
            current: List[Tuple[str, str]] = merge_intervals_from_candidates(groups[0])
            for group in groups[1:]:
                right = merge_intervals_from_candidates(group)
                out: List[Tuple[str, str]] = []
                i, j = 0, 0
                while i < len(current) and j < len(right):
                    left_s, left_e = current[i]
                    right_s, right_e = right[j]
                    s = max(left_s, right_s)
                    e = min(left_e, right_e)
                    if s <= e:
                        out.append((s, e))
                    if left_e < right_e:
                        i += 1
                    else:
                        j += 1
                # normalize overlaps after each pairwise intersection
                if out:
                    out_candidates = [
                        Candidate("", "", "", s, e, f"{s}-{e}") for s, e in out
                    ]
                    current = merge_intervals_from_candidates(out_candidates)
                else:
                    current = []
                    break
            return [f"{s} to {e}" for s, e in current], debug

        # Duration operators sum/average:
        # duration of each logical seed SRO is computed after interval merging,
        # avoiding double counting overlapping or duplicate KG facts.
        group_values = [group_duration_days_from_candidates(group) for group in groups]
        debug["group_duration_days"] = group_values
        if operator == "sum":
            return [f"{sum(group_values)} days"], debug
        return [average_duration_value(group_values, mode=duration_average_mode)], debug

    # Option/rank-sensitive operators: keep deterministic source-event semantics.
    ans, source_dbg = answer_operator_from_events(
        case,
        events,
        operator,
        duration_tie_policy=duration_tie_policy,
    )
    debug["mode"] = "source_event_operator_semantics_preserved"
    debug["operator_debug"] = source_dbg
    debug["semantic_gold_events"] = [e.raw for e in sorted(events, key=_event_sort_key)]
    return ans, debug


# ----------------------------- operator answers from events -----------------------------

def union_intervals(events: Sequence[Event]) -> List[str]:
    if not events:
        return []
    intervals = sorted([(e.start, e.end) for e in events], key=lambda x: x[0])
    merged: List[Tuple[str, str]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return [f"{s} to {e}" for s, e in merged]


def intersection_intervals(events: Sequence[Event]) -> List[str]:
    if not events:
        return []
    starts = [parse_date_iso(e.start) for e in events]
    ends = [parse_date_iso(e.end) for e in events]
    s = max(starts)
    e = min(ends)
    if s > e:
        return []
    return [f"{s.isoformat()} to {e.isoformat()}"]


def source_event_answer_from_event(e: Event, answer_type: str) -> List[str]:
    if answer_type == "subject":
        return [e.head]
    if answer_type == "object":
        return [e.tail]
    if answer_type in {"timestamp", "timestamp_start"}:
        return [e.start]
    if answer_type == "timestamp_end":
        return [e.end]

    if answer_type in {"timestamp_range"}:
    # if answer_type in {"timestamp_range", "duration"}:
        return [f"{e.start} to {e.end}"]
    if answer_type == "duration":
        return [format_duration_answer(e.start, e.end)]
    return []


def answer_operator_from_events(case: Dict[str, Any], events: List[Event], operator: str, duration_tie_policy: str = "all") -> Tuple[List[str], Dict[str, Any]]:
    answer_type = case.get("answer_type", "")
    question = case.get("question", "")
    debug: Dict[str, Any] = {}

    if operator == "timeline":
        if len(events) != 1:
            debug["warning"] = "timeline_with_multiple_events"
        ordered_events = sorted(events, key=_event_sort_key)
        answers: List[str] = []
        for e in ordered_events:
            answers.extend(source_event_answer_from_event(e, answer_type))
        return dedup_keep_order(answers), debug

    if operator == "union":
        return union_intervals(events), debug

    if operator == "intersection":
        return intersection_intervals(events), debug

    if operator == "sum":
        return [f"{sum(duration_days(e) for e in events)} days"], debug

    if operator == "average":
        if not events:
            return [], debug
        return [f"{math.floor(sum(duration_days(e) for e in events) / len(events))} days"], debug

    if operator == "duration":
        # In this dataset plain duration is interval, not N days.
        # return [interval_str_from_event(e) for e in sorted(events, key=_event_sort_key)], debug

        ordered_events = sorted(events, key=_event_sort_key)
        return [
            format_duration_answer(e.start, e.end)
            for e in ordered_events
        ], debug

    if operator == "duration_compare":
        # duration_compare has three subtypes in this dataset.
        #
        # A) "Which one lasted the longest/shortest among A, B, C?"
        #    Return the 1-based option number(s) according to QUESTION order.
        #    If there is a tie, duration_tie_policy controls behavior:
        #      - all: return all tied option numbers
        #      - first: return first tied option in question order
        #      - exception: return [] and mark tie ambiguous
        #
        # B) "Which one is the 2nd longest/shortest among A, B, C?"
        #    Return the 1-based option number at that ordinal position after
        #    sorting by duration and breaking ties by question order.
        #
        # C) "Is the duration of A longer/shorter/equal to the duration of B?"
        #    Compare the first-mentioned event with the second-mentioned event and
        #    return longer / shorter / equals.
        subtype = duration_compare_subtype(question)
        q = question.lower()
        question_order = event_indices_in_question_order(question, events)
        durations_by_event_idx = [duration_days(e) for e in events]
        order_rank = {idx: pos for pos, idx in enumerate(question_order)}

        if subtype == "which_one_ranked_longest_shortest":
            parsed_rank = parse_ranked_duration_compare(question)
            if parsed_rank is None:
                return [], {
                    "warning": "ranked_duration_compare_parse_failed",
                    "subtype": subtype,
                    "question": question,
                    "durations_by_event_idx": durations_by_event_idx,
                    "question_order_indices": question_order,
                }

            rank_n, longest_or_shortest = parsed_rank
            if rank_n < 1 or rank_n > len(events):
                return [], {
                    "warning": "ranked_duration_compare_rank_out_of_range",
                    "subtype": subtype,
                    "rank_n": rank_n,
                    "event_count": len(events),
                    "durations_by_event_idx": durations_by_event_idx,
                    "question_order_indices": question_order,
                }

            reverse = longest_or_shortest == "longest"
            if reverse:
                sorted_indices = sorted(
                    range(len(events)),
                    key=lambda i: (-durations_by_event_idx[i], order_rank.get(i, i)),
                )
            else:
                sorted_indices = sorted(
                    range(len(events)),
                    key=lambda i: (durations_by_event_idx[i], order_rank.get(i, i)),
                )

            target_idx = sorted_indices[rank_n - 1]
            option_number = order_rank.get(target_idx, target_idx) + 1

            return [str(option_number)], {
                "subtype": subtype,
                "rank_n": rank_n,
                "longest_or_shortest": longest_or_shortest,
                "durations_by_event_idx": durations_by_event_idx,
                "question_order_indices": question_order,
                "sorted_event_indices": sorted_indices,
                "target_event_index": target_idx,
                "option_number": option_number,
                "answer_policy": "ranked_duration_index_by_question_order",
            }

        if subtype == "which_one_longest_shortest":
            if not events:
                return [], {"warning": "duration_compare_no_events", "subtype": subtype}

            if "shortest" in q:
                target_duration = min(durations_by_event_idx)
                tied_indices = [i for i, d in enumerate(durations_by_event_idx) if d == target_duration]
            else:
                target_duration = max(durations_by_event_idx)
                tied_indices = [i for i, d in enumerate(durations_by_event_idx) if d == target_duration]

            tied_indices = sorted(tied_indices, key=lambda i: order_rank.get(i, i))

            if len(tied_indices) > 1:
                if duration_tie_policy == "exception":
                    return [], {
                        "warning": "duration_compare_tie_ambiguous",
                        "subtype": subtype,
                        "tie_policy": duration_tie_policy,
                        "target_duration": target_duration,
                        "tied_event_indices": tied_indices,
                        "tied_option_numbers": [order_rank.get(i, i) + 1 for i in tied_indices],
                        "durations_by_event_idx": durations_by_event_idx,
                        "question_order_indices": question_order,
                    }
                if duration_tie_policy == "first":
                    target_idx = tied_indices[0]
                    option_number = order_rank.get(target_idx, target_idx) + 1
                    return [str(option_number)], {
                        "subtype": subtype,
                        "tie_policy": duration_tie_policy,
                        "tie_detected": True,
                        "target_duration": target_duration,
                        "tied_event_indices": tied_indices,
                        "tied_option_numbers": [order_rank.get(i, i) + 1 for i in tied_indices],
                        "durations_by_event_idx": durations_by_event_idx,
                        "question_order_indices": question_order,
                        "target_event_index": target_idx,
                        "option_number": option_number,
                        "answer_policy": "which_one_tie_first_index_by_question_order",
                    }
                # default: all
                option_numbers = [str(order_rank.get(i, i) + 1) for i in tied_indices]
                return option_numbers, {
                    "subtype": subtype,
                    "tie_policy": duration_tie_policy,
                    "tie_detected": True,
                    "target_duration": target_duration,
                    "tied_event_indices": tied_indices,
                    "tied_option_numbers": option_numbers,
                    "durations_by_event_idx": durations_by_event_idx,
                    "question_order_indices": question_order,
                    "answer_policy": "which_one_tie_all_indices_by_question_order",
                }

            target_idx = tied_indices[0]
            option_number = order_rank.get(target_idx, target_idx) + 1
            return [str(option_number)], {
                "subtype": subtype,
                "tie_policy": duration_tie_policy,
                "tie_detected": False,
                "target_duration": target_duration,
                "durations_by_event_idx": durations_by_event_idx,
                "question_order_indices": question_order,
                "target_event_index": target_idx,
                "option_number": option_number,
                "answer_policy": "which_one_index_by_question_order",
            }

        if subtype == "binary_longer_shorter_equal":
            if len(events) != 2:
                return [], {
                    "warning": "binary_duration_compare_requires_two_events",
                    "subtype": subtype,
                    "event_count": len(events),
                    "question_order_indices": question_order,
                }

            if len(question_order) >= 2:
                i0, i1 = question_order[0], question_order[1]
            else:
                i0, i1 = 0, 1

            d0, d1 = duration_days(events[i0]), duration_days(events[i1])
            if d0 > d1:
                ans = "longer"
            elif d0 < d1:
                ans = "shorter"
            else:
                ans = "equals"

            return [ans], {
                "subtype": subtype,
                "durations_by_event_idx": durations_by_event_idx,
                "question_order_indices": question_order,
                "left_event_index": i0,
                "right_event_index": i1,
                "left_duration": d0,
                "right_duration": d1,
                "answer_policy": "binary_duration_relation_by_question_order",
            }

        return [], {
            "warning": "unsupported_duration_compare_subtype",
            "subtype": subtype,
            "question": question,
            "durations_by_event_idx": durations_by_event_idx,
            "question_order_indices": question_order,
        }

    if operator in {"rank_start_time", "rank_end_time"}:
        target_idx = find_rank_target_index(question, events)
        if target_idx is None:
            return [], {"warning": "rank_target_not_found"}
        if operator == "rank_start_time":
            ordered = sorted(range(len(events)), key=lambda i: events[i].start)
        else:
            ordered = sorted(range(len(events)), key=lambda i: events[i].end)
        return [str(ordered.index(target_idx) + 1)], {"target_idx": target_idx, "ordered_indices": ordered}

    return [], {"warning": f"unsupported_operator:{operator}"}


# ----------------------------- case compilation and execution -----------------------------

def compile_single_offset_case(case: Dict[str, Any], events: List[Event], parsed: Dict[str, Any]) -> Dict[str, Any]:
    question = case.get("question", "")
    answer_type = case.get("answer_type", "")
    offset = extract_offset_from_question(question)
    if offset is None:
        return {"status": "OFFSET_NOT_FOUND"}
    offset_days, surface_direction = offset

    offset_clause, main_clause = split_single_offset_question(question)
    offset_scores = [event_score_in_text(offset_clause, e) for e in events]
    main_scores = [event_score_in_text(main_clause, e) for e in events]

    if not events:
        return {"status": "NO_EVENTS"}

    query_idx = max(range(len(events)), key=lambda i: main_scores[i])
    anchor_idx = max(range(len(events)), key=lambda i: offset_scores[i])

    status = "OK"
    hidden_anchor = False
    if query_idx == anchor_idx and len(events) == 2:
        other = 1 - query_idx
        if event_score_in_text(question, events[other]) == 0:
            anchor_idx = other
            hidden_anchor = True
            status = "OK_WITH_HIDDEN_ANCHOR"

    constraint = dict(parsed["constraints"][0])
    constraint["offset_days"] = offset_days
    constraint["surface_direction"] = surface_direction

    query_template, projection = build_query_template(events[query_idx], answer_type)
    return {
        "status": status,
        "program_type": "single_offset",
        "query_idx": query_idx,
        "anchor_indices": [anchor_idx],
        "query_template": query_template,
        "projection": projection,
        "constraints": [constraint],
        "debug": {
            "offset_clause": offset_clause,
            "main_clause": main_clause,
            "offset_scores": offset_scores,
            "main_scores": main_scores,
            "hidden_anchor": hidden_anchor,
        },
    }


def compile_constraint_case(case: Dict[str, Any], events: List[Event], parsed: Dict[str, Any]) -> Dict[str, Any]:
    question = case.get("question", "")
    answer_type = case.get("answer_type", "")
    constraints = parsed["constraints"]
    expected = len(constraints)

    anchor_indices, link_debug = link_anchors_from_question(question, events, expected)
    if len(anchor_indices) != expected:
        return {
            "status": "ANCHOR_LINK_FAILED",
            "program_type": "constraint",
            "anchor_indices": anchor_indices,
            "debug": link_debug,
        }

    query_idx = infer_query_index(events, anchor_indices)
    if query_idx is None:
        return {
            "status": "QUERY_INFER_FAILED",
            "program_type": "constraint",
            "anchor_indices": anchor_indices,
            "debug": link_debug,
        }

    query_template, projection = build_query_template(events[query_idx], answer_type)
    return {
        "status": "OK",
        "program_type": "constraint",
        "query_idx": query_idx,
        "anchor_indices": anchor_indices,
        "query_template": query_template,
        "projection": projection,
        "constraints": constraints,
        "debug": link_debug,
    }


def compile_case(case: Dict[str, Any]) -> Dict[str, Any]:
    events = [parse_event(x) for x in case.get("events", [])]
    parsed = parse_temporal_relation(case.get("temporal_relation"))

    if parsed["unsupported"]:
        return {
            "status": "UNSUPPORTED_TEMPORAL_RELATION",
            "program_type": "unsupported",
            "parsed_temporal_relation": parsed,
        }

    if parsed["operator"] is not None:
        return {
            "status": "OK",
            "program_type": "operator",
            "operator": parsed["operator"],
            "parsed_temporal_relation": parsed,
        }

    # duration_after / duration_before: one constraint with needs_question_offset.
    if len(parsed["constraints"]) == 1 and parsed["constraints"][0].get("needs_question_offset"):
        out = compile_single_offset_case(case, events, parsed)
        out["parsed_temporal_relation"] = parsed
        return out

    if parsed["constraints"]:
        out = compile_constraint_case(case, events, parsed)
        out["parsed_temporal_relation"] = parsed
        return out

    return {"status": "NO_PROGRAM", "program_type": "none", "parsed_temporal_relation": parsed}



def execute_compiled_case(
    case: Dict[str, Any],
    compiled: Dict[str, Any],
    graph: Any = None,
    graph_index: Optional[GraphExactIndex] = None,
    duration_tie_policy: str = "all",
) -> Tuple[List[str], Dict[str, Any]]:
    events = [parse_event(x) for x in case.get("events", [])]
    answer_type = case.get("answer_type", "")

    if compiled.get("status") not in {"OK", "OK_WITH_HIDDEN_ANCHOR"}:
        return [], {"reason": "compile_not_ok"}

    ptype = compiled.get("program_type")

    if ptype == "operator":
        return execute_operator_with_graph(
            case,
            events,
            compiled["operator"],
            graph=graph,
            graph_index=graph_index,
            duration_tie_policy=duration_tie_policy,
            duration_average_mode="floor",
        )

    query_idx = compiled.get("query_idx")
    if query_idx is None:
        return [], {"reason": "no_query_idx"}

    # source-event gold from events only.
    source_gold = source_event_answer_from_event(events[query_idx], answer_type)

    # If graph/index is not given, return source-event answer only.
    if graph is None and graph_index is None:
        return source_gold, {
            "mode": "source_event_no_graph",
            "source_event_gold": source_gold,
            "semantic_gold_events": [events[query_idx].raw],
        }

    query_template = compiled["query_template"]
    candidates, scan_mode, query_notes = scan_semantic_candidates(
        graph=graph,
        graph_index=graph_index,
        query=query_template,
    )

    # Apply constraints over anchors from events.
    filtered = candidates
    for c in compiled.get("constraints", []):
        ai = c["anchor_index"]
        anchor_event_index = compiled["anchor_indices"][ai]
        anchor_event = events[anchor_event_index]
        filtered = [
            cand for cand in filtered
            if match_constraint(cand, anchor_event, c, question=str(case.get("question", "") or ""))
        ]

    projection = compiled["projection"]
    answers = project_candidates_sorted_by_start(filtered, projection, answer_type)
    answer_event_strings = semantic_gold_events_from_candidates(
        filtered,
        projection=projection,
        answer_type=answer_type,
        semantic_gold=answers,
    )

    return answers, {
        "mode": scan_mode,
        "query_notes": query_notes,
        "candidate_count": len(candidates),
        "filtered_count": len(filtered),
        "source_event_gold": source_gold,
        "semantic_gold_order": "candidate_start_time_ascending",
        "semantic_gold_events": answer_event_strings,
        "semantic_gold_events_order": "answer_events_start_time_ascending",
    }


def compare_answers(original: Any, repaired: Any) -> str:
    o = set(normalize_answer_values(original))
    r = set(normalize_answer_values(repaired))
    if not r:
        return "NO_REPAIRED_ANSWER"
    if o == r:
        return "ORIGINAL_EXACT"
    if o and o.issubset(r):
        return "ORIGINAL_SUBSET_OF_REPAIRED"
    if r.issubset(o):
        return "REPAIRED_SUBSET_OF_ORIGINAL"
    return "DIFFERENT"

def all_events_are_point_events(case: Dict[str, Any]) -> bool:
    """Return True iff every event in case["events"] has start == end.

    For answer_type == relation_duration, all-point cases are routed to
    exceptions instead of being accepted into the clean dataset.
    """
    event_strs = case.get("events", []) or []
    if not event_strs:
        return False

    parsed: List[Event] = []
    for ev in event_strs:
        try:
            parsed.append(parse_event(ev))
        except Exception:
            return False

    return bool(parsed) and all(e.start == e.end for e in parsed)



def repair_one_case(
    case: Dict[str, Any],
    graph: Any = None,
    graph_index: Optional[GraphExactIndex] = None,
    duration_tie_policy: str = "all",
) -> Dict[str, Any]:
    """Repair one case from raw dataset fields only.

    Repair order:
      1. prune seed events whose head and tail are both absent from question;
      2. normalize simple timeline-duration original answers from interval form
         to the numeric-duration convention "N days";
      3. compile the temporal program from the normalized working case;
      4. execute deterministic semantic repair, optionally against the KG;
      5. compare normalized original answer vs semantic gold.

    Important:
      All downstream work uses the same `working_case`. This prevents the
      original-answer normalization from being applied to one object while the
      semantic repair is executed on another object.
    """
    # 1. Remove obviously ungrounded seed events first.
    working_case, dropped_events = prune_unmentioned_events_from_case(case)

    # 2. Normalize simple-duration original answers before semantic repair.
    #    This allows answer comparison to use the same target convention as
    #    semantic_gold generation: "N days" rather than interval strings.
    working_case = normalize_simple_duration_original_answer(working_case)

    # 3. Use the normalized working case consistently from here onward.
    new_case = dict(working_case)
    original_answer = normalize_answer_values(working_case.get("answer", []))

    try:
        compiled = compile_case(working_case)
        repaired_answer, exec_debug = execute_compiled_case(
            working_case,
            compiled,
            graph=graph,
            graph_index=graph_index,
            duration_tie_policy=duration_tie_policy,
        )
        exec_debug = dict(exec_debug or {})
        exec_debug["pruned_unmentioned_events"] = dropped_events
        exec_debug["pruned_unmentioned_event_count"] = len(dropped_events)

        new_case["repaired"] = {
            "program_status": compiled.get("status"),
            "program_type": compiled.get("program_type"),
            "compiled": compiled,
            "semantic_gold": repaired_answer,
            "original_answer": original_answer,
            "repair_status": compare_answers(original_answer, repaired_answer),
            "exec_debug": exec_debug,
        }
    except Exception as e:
        new_case["repaired"] = {
            "program_status": "EXCEPTION",
            "program_type": None,
            "semantic_gold": [],
            "original_answer": original_answer,
            "repair_status": "EXCEPTION",
            "error": repr(e),
            "exec_debug": {
                "pruned_unmentioned_events": dropped_events,
                "pruned_unmentioned_event_count": len(dropped_events),
            },
        }

    return new_case

def repair_dataset(
    data: List[Dict[str, Any]],
    graph: Any = None,
    graph_index: Optional[GraphExactIndex] = None,
    show_progress: bool = True,
    workers: int = 1,
    duration_tie_policy: str = "all",
) -> List[Dict[str, Any]]:
    workers = max(1, int(workers or 1))
    total = len(data)

    if total == 0:
        return []

    if workers == 1:
        out = []
        iterator = iter_with_progress(
            data,
            total=total,
            desc="Repairing gold",
            enabled=show_progress,
        )
        for case in iterator:
            out.append(
                repair_one_case(
                    case,
                    graph=graph,
                    graph_index=graph_index,
                    duration_tie_policy=duration_tie_policy,
                )
            )
        return out

    out: List[Optional[Dict[str, Any]]] = [None] * total

    def _worker(idx: int, case: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        repaired_case = repair_one_case(
            case,
            graph=graph,
            graph_index=graph_index,
            duration_tie_policy=duration_tie_policy,
        )
        return idx, repaired_case


    max_pending = max(workers * 4, workers)

    data_iter = iter(enumerate(data))
    pending = set()

    with ThreadPoolExecutor(max_workers=workers) as ex:

        for _ in range(min(max_pending, total)):
            idx, case = next(data_iter)
            pending.add(ex.submit(_worker, idx, case))

        iterator = range(total)
        if show_progress and tqdm is not None:
            progress = tqdm(
                total=total,
                desc=f"Repairing gold ({workers} threads)",
                unit="case",
            )
        else:
            progress = None

        completed = 0

        while pending:
            done, pending = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            for fut in done:
                idx, repaired_case = fut.result()
                out[idx] = repaired_case
                completed += 1

                if progress is not None:
                    progress.update(1)

                try:
                    next_idx, next_case = next(data_iter)
                    pending.add(ex.submit(_worker, next_idx, next_case))
                except StopIteration:
                    pass

        if progress is not None:
            progress.close()

    return [x for x in out if x is not None]



def summarize(repaired: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter

    c_status = Counter()
    c_type = Counter()
    c_repair = Counter()
    c_temporal = Counter()
    c_duration_compare_subtype = Counter()
    c_scan_mode = Counter()
    pruned_event_total = 0
    cases_with_pruned_events = 0

    for x in repaired:
        r = x.get("repaired", {})
        c_status[r.get("program_status")] += 1
        c_type[r.get("program_type")] += 1
        c_repair[r.get("repair_status")] += 1
        c_temporal[x.get("temporal_relation")] += 1

        exec_dbg = r.get("exec_debug", {}) or {}
        pruned_n = int(exec_dbg.get("pruned_unmentioned_event_count", 0) or 0)
        pruned_event_total += pruned_n
        if pruned_n > 0:
            cases_with_pruned_events += 1

        mode = exec_dbg.get("mode")
        if mode:
            c_scan_mode[str(mode)] += 1

        if x.get("temporal_relation") == "duration_compare":
            dbg = exec_dbg.get("operator_debug", {})
            c_duration_compare_subtype[dbg.get("subtype", "missing_subtype")] += 1

    summary = {
        "n": len(repaired),
        "program_status": dict(c_status.most_common()),
        "program_type": dict(c_type.most_common()),
        "repair_status": dict(c_repair.most_common()),
        "top_temporal_relation": dict(c_temporal.most_common(30)),
        "pruned_unmentioned_events": {
            "cases": cases_with_pruned_events,
            "events": pruned_event_total,
        },
        "execution_mode": dict(c_scan_mode.most_common()),
    }
    if c_duration_compare_subtype:
        summary["duration_compare_subtype"] = dict(c_duration_compare_subtype.most_common())
    return summary

def split_repaired_cases(
    repaired: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split cases into exact and review groups.

    exact_cases: repair_status == ORIGINAL_EXACT.
    review_cases: all non-exact cases, including subset, different, no repaired answer, and exceptions.
    """
    exact_cases: List[Dict[str, Any]] = []
    review_cases: List[Dict[str, Any]] = []

    for case in repaired:
        status = case.get("repaired", {}).get("repair_status")
        if status == "ORIGINAL_EXACT":
            exact_cases.append(case)
        else:
            review_cases.append(case)

    return exact_cases, review_cases



def _semantic_gold(case: Dict[str, Any]) -> List[str]:
    return list(case.get("repaired", {}).get("semantic_gold", []) or [])



def should_keep_for_clean_dataset(case: Dict[str, Any], max_subset_gold_size: int = 10) -> Tuple[bool, str]:
    """Decide whether a repaired case enters the SCoP-ready clean dataset.

    Accepted:
      - ORIGINAL_EXACT
      - ORIGINAL_SUBSET_OF_REPAIRED, when repaired answer size is bounded
      - REPAIRED_SUBSET_OF_ORIGINAL, because this is the expected signature
        of removing polluted events / over-wide original answers

    Rejected to exceptions:
      - DIFFERENT
      - NO_REPAIRED_ANSWER
      - EXCEPTION / unsupported compile outcomes
    """
    repaired = case.get("repaired", {})
    status = repaired.get("repair_status")
    semantic = normalize_answer_values(repaired.get("semantic_gold", []) or [])

    if status == "ORIGINAL_EXACT":
        return True, "ORIGINAL_EXACT"

    if status == "ORIGINAL_SUBSET_OF_REPAIRED" and len(semantic) <= max_subset_gold_size:
        return True, f"ORIGINAL_SUBSET_OF_REPAIRED_SIZE_LE_{max_subset_gold_size}"

    if status == "ORIGINAL_SUBSET_OF_REPAIRED":
        return False, f"SUBSET_BUT_SEMANTIC_GOLD_TOO_LARGE:{len(semantic)}>{max_subset_gold_size}"

    if status == "REPAIRED_SUBSET_OF_ORIGINAL" and semantic:
        return True, "REPAIRED_SUBSET_OF_ORIGINAL_ACCEPTED"

    return False, f"REPAIR_STATUS_NOT_ACCEPTED:{status}"

def _safe_parse_events(event_strs: Sequence[str]) -> List[Event]:
    out: List[Event] = []
    for s in event_strs or []:
        try:
            out.append(parse_event(s))
        except Exception:
            continue
    return out



def format_events_query_then_anchor(case: Dict[str, Any]) -> List[str]:
    """Construct repaired evidence events for the clean dataset.

    Ordering contract for SCoP downstream repair scripts:

    Constraint / single-offset cases:
      1. preserve the original source query/search event first;
      2. place anchor events after the query, in compiled anchor order;
      3. append KG-completed semantic answer/query support events after those;
      4. if the source query is also present in semantic support events, dedupe it.

    This preserves the historical dataset convention relied on by
    ``4repair_duration.py``:
      - medium duration_before / duration_after:
            events[0] = query/search event
            events[1] = anchor event
      - complex subject-prefix repair:
            events[0] = query/search event

    Operator cases:
      1. use KG-completed semantic support events when available;
      2. otherwise fall back to sorted pruned source events.

    We intentionally do NOT append arbitrary leftover original events: those were
    a source of evidence pollution in earlier repair variants.
    """
    original_events = list(case.get("events", []) or [])
    compiled = case.get("repaired", {}).get("compiled", {}) or {}
    exec_debug = case.get("repaired", {}).get("exec_debug", {}) or {}

    if not original_events:
        return []

    support_event_strings = list(exec_debug.get("semantic_gold_events", []) or [])
    program_type = compiled.get("program_type")

    # Operator cases: use semantic KG support events whenever available.
    if program_type == "operator":
        if support_event_strings:
            return dedup_keep_order(support_event_strings)
        events = _safe_parse_events(original_events)
        if len(events) == len(original_events):
            return [e.raw for e in sorted(events, key=_event_sort_key)]
        return original_events

    anchor_indices = compiled.get("anchor_indices")
    query_idx = compiled.get("query_idx")

    if isinstance(anchor_indices, list) or query_idx is not None:
        ordered: List[str] = []
        seen: set[str] = set()

        # 1) Original source query/search event first.
        if isinstance(query_idx, int) and 0 <= query_idx < len(original_events):
            ev = original_events[query_idx]
            if ev not in seen:
                ordered.append(ev)
                seen.add(ev)

        # 2) Anchor events next, in compiled anchor order.
        if isinstance(anchor_indices, list):
            for idx in anchor_indices:
                if isinstance(idx, int) and 0 <= idx < len(original_events):
                    ev = original_events[idx]
                    if ev not in seen:
                        ordered.append(ev)
                        seen.add(ev)

        # 3) KG-completed semantic query/answer support events afterward.
        if support_event_strings:
            for ev in support_event_strings:
                if ev not in seen:
                    ordered.append(ev)
                    seen.add(ev)

        return ordered

    events = _safe_parse_events(original_events)
    if len(events) == len(original_events):
        return [e.raw for e in sorted(events, key=_event_sort_key)]
    return original_events

def build_clean_and_exception_datasets(
    repaired: List[Dict[str, Any]],
    max_subset_gold_size: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean: List[Dict[str, Any]] = []
    exceptions: List[Dict[str, Any]] = []
    for case in repaired:
        keep, reason = should_keep_for_clean_dataset(case, max_subset_gold_size=max_subset_gold_size)
        semantic = _semantic_gold(case)
        if keep:
            new_case = dict(case)
            new_repaired = dict(new_case.get("repaired", {}) or {})
            new_repaired["clean_filter"] = {
                "kept": True,
                "reason": reason,
                "max_subset_gold_size": max_subset_gold_size,
                "answer_replaced_with": "repaired.semantic_gold",
                "events_policy": "query_first_then_anchors_then_semantic_support_events_for_constraint_cases;start_time_sorted_for_operator_cases",
            }
            new_case["repaired"] = new_repaired
            new_case["original_answer"] = normalize_answer_values(case.get("answer", []) or [])
            new_case["answer"] = semantic
            new_case["answer_policy"] = "semantic_gold_verified"
            new_case["events"] = format_events_query_then_anchor(case)
            clean.append(new_case)
        else:
            ex_case = dict(case)
            ex_repaired = dict(ex_case.get("repaired", {}) or {})
            ex_repaired["clean_filter"] = {
                "kept": False,
                "reason": reason,
                "max_subset_gold_size": max_subset_gold_size,
            }
            ex_case["repaired"] = ex_repaired
            exceptions.append(ex_case)
    return clean, exceptions


def default_clean_exception_paths(output_path: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".json"
    return f"{base}.clean{ext}", f"{base}.exceptions{ext}"


def default_core_clean_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".json"
    return f"{base}.clean.core{ext}"


def build_core_clean_dataset(clean_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Export compact clean cases with only the requested fields.

    Input must be the already-filtered clean cases, where answer has already
    been replaced by semantic_gold and events have already been reordered.
    """
    fields = [
        "id",
        "question",
        "answer",
        "events",
        "question_level",
        "question_type",
        "answer_type",
        "temporal_relation",
        "split",
    ]
    out: List[Dict[str, Any]] = []
    for case in clean_cases:
        row = {k: case.get(k) for k in fields}
        if row.get("answer") is None:
            row["answer"] = []
        if row.get("events") is None:
            row["events"] = []
        out.append(row)
    return out


def default_split_paths(output_path: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".json"
    return f"{base}.exact{ext}", f"{base}.review{ext}"

from collections import Counter, defaultdict


def summarize_repair_by_question_level(
    repaired: List[Dict[str, Any]],
    clean_cases: List[Dict[str, Any]],
    exception_cases: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Summarize repair results by question_level.

    Output dimensions:
      - total repaired inputs
      - kept / dropped after clean filtering
      - repair_status distribution
      - program_status distribution
      - program_type distribution
    """
    levels = ("simple", "medium", "complex")

    result: Dict[str, Dict[str, Any]] = {
        level: {
            "total": 0,
            "kept_clean": 0,
            "dropped_exception": 0,
            "repair_status": Counter(),
            "program_status": Counter(),
            "program_type": Counter(),
        }
        for level in levels
    }

    # -------------------------
    # Overall repaired statistics
    # -------------------------
    for case in repaired:
        level = str(case.get("question_level", "unknown"))
        if level not in result:
            result[level] = {
                "total": 0,
                "kept_clean": 0,
                "dropped_exception": 0,
                "repair_status": Counter(),
                "program_status": Counter(),
                "program_type": Counter(),
            }

        repaired_meta = case.get("repaired", {}) or {}

        result[level]["total"] += 1
        result[level]["repair_status"][repaired_meta.get("repair_status")] += 1
        result[level]["program_status"][repaired_meta.get("program_status")] += 1
        result[level]["program_type"][repaired_meta.get("program_type")] += 1

    # -------------------------
    # Clean / exception counts
    # -------------------------
    for case in clean_cases:
        level = str(case.get("question_level", "unknown"))
        if level not in result:
            result[level] = {
                "total": 0,
                "kept_clean": 0,
                "dropped_exception": 0,
                "repair_status": Counter(),
                "program_status": Counter(),
                "program_type": Counter(),
            }
        result[level]["kept_clean"] += 1

    for case in exception_cases:
        level = str(case.get("question_level", "unknown"))
        if level not in result:
            result[level] = {
                "total": 0,
                "kept_clean": 0,
                "dropped_exception": 0,
                "repair_status": Counter(),
                "program_status": Counter(),
                "program_type": Counter(),
            }
        result[level]["dropped_exception"] += 1

    # -------------------------
    # Counter -> dict
    # -------------------------
    final_result: Dict[str, Dict[str, Any]] = {}

    for level, payload in result.items():
        final_result[level] = {
            "total": payload["total"],
            "kept_clean": payload["kept_clean"],
            "dropped_exception": payload["dropped_exception"],
            "repair_status": dict(payload["repair_status"].most_common()),
            "program_status": dict(payload["program_status"].most_common()),
            "program_type": dict(payload["program_type"].most_common()),
        }

    return final_result


def normalize_simple_duration_original_answer(
    case: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert old simple-duration interval gold answers into the new
    numeric-duration convention before semantic repair/comparison.

    Dataset convention:
        duration_days = end_date - start_date
        point event such as 1900-01-01 to 1900-01-01 = 0 days
    """
    if not is_simple_duration_timeline_case(case):
        return case

    event_strs = case.get("events", []) or []
    if not event_strs:
        return case

    try:
        events = [parse_event(x) for x in event_strs]
    except Exception:
        return case

    if not events:
        return case


    normalized_answer: List[str] = []
    for e in events:
        normalized_answer.append(
            format_duration_answer(e.start, e.end)
        )

    normalized_answer = dedup_keep_order(normalized_answer)

    out = dict(case)


    old_answer = normalize_answer_values(case.get("answer", []))
    if old_answer != normalized_answer:
        out["original_answer_before_duration_normalization"] = old_answer
        out["answer"] = normalized_answer

    return out



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input dataset json")
    ap.add_argument("--output", required=True, help="Path to final repaired dataset json")
    ap.add_argument(
        "--graph",
        default=None,
        help="Optional igraph pickle path. Strongly recommended for semantic gold repair.",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Optional repair report JSON path. Default: <output>.report.json",
    )
    ap.add_argument(
        "--max-subset-gold-size",
        type=int,
        default=10,
        help="Keep subset-expanded cases only when len(semantic_gold) <= this value. Default: 10",
    )
    ap.add_argument(
        "--duration-tie-policy",
        choices=["all", "first", "exception"],
        default="all",
        help="Tie handling for duration_compare. Default: all",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads for case-level repair. Default: 1",
    )
    ap.add_argument(
        "--no-index",
        action="store_true",
        help="Disable prebuilt exact KG edge index and use slow graph scan instead",
    )
    args = ap.parse_args()

    data = load_json(args.input)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of cases.")

    graph = None
    graph_index = None

    if args.graph:
        graph = load_igraph_pickle(args.graph)
        if not args.no_index:
            graph_index = build_graph_exact_index(
                graph,
                show_progress=not args.no_progress,
            )

    # --------------------------------------------------
    # 1. Perform repair in memory
    # --------------------------------------------------
    repaired = repair_dataset(
        data,
        graph=graph,
        graph_index=graph_index,
        show_progress=not args.no_progress,
        workers=args.workers,
        duration_tie_policy=args.duration_tie_policy,
    )

    # --------------------------------------------------
    # 2. Keep only accepted clean cases
    # --------------------------------------------------
    clean_cases, exception_cases = build_clean_and_exception_datasets(
        repaired,
        max_subset_gold_size=args.max_subset_gold_size,
    )

    # --------------------------------------------------
    # 3. Export compact dataset directly as final output
    # --------------------------------------------------
    final_cases = build_core_clean_dataset(clean_cases)
    save_json(args.output, final_cases)

    # --------------------------------------------------
    # 4. Build lightweight report
    # --------------------------------------------------
    summ = summarize(repaired)
    level_summary = summarize_repair_by_question_level(
    repaired=repaired,
    clean_cases=clean_cases,
    exception_cases=exception_cases,
)
    summ["question_level_repair_summary"] = level_summary
    summ["final_output"] = {
        "output_path": args.output,
        "input_count": len(data),
        "kept_count": len(final_cases),
        "dropped_count": len(exception_cases),
    }
    summ["clean_filter"] = {
        "policy": (
            "keep ORIGINAL_EXACT; "
            "keep ORIGINAL_SUBSET_OF_REPAIRED when len(semantic_gold) <= max_subset_gold_size; "
            "keep non-empty REPAIRED_SUBSET_OF_ORIGINAL"
        ),
        "max_subset_gold_size": args.max_subset_gold_size,
        "clean_count": len(clean_cases),
        "exception_count": len(exception_cases),
    }

    report_path = args.report or (args.output + ".report.json")
    save_json(report_path, summ)

    print("=" * 96)
    print("[CornQuestionsKG Repair Report]")
    print(f"Input:         {args.input}")
    print(f"Output:        {args.output}")
    print(f"Report:        {report_path}")
    print(f"Input count:   {len(data)}")
    print(f"Kept count:    {len(final_cases)}")
    print(f"Dropped count: {len(exception_cases)}")
    print("=" * 96)
    # print(json.dumps(summ, ensure_ascii=False, indent=2))
    
    print("=" * 96)
    print("[Repair Summary by question_level]")
    for level in ("simple", "medium", "complex"):
        info = level_summary.get(level, {})
        print(
            f"{level:<8} "
            f"total={info.get('total', 0):<6} "
            f"kept={info.get('kept_clean', 0):<6} "
            f"dropped={info.get('dropped_exception', 0):<6}"
        )

    print("=" * 96)


if __name__ == "__main__":
    main()
