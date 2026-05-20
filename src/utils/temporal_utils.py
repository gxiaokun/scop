import re
import calendar
from copy import deepcopy
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from src.config.base_config import logger



_ORDINAL_SUFFIX_RE = re.compile(r"(\d+)(st|nd|rd|th)", re.IGNORECASE)

def sort_candidates_by_time(
    candidates: List[Dict[str, Any]],
    reverse: bool = False,
) -> List[Dict[str, Any]]:

    enriched = []
    for c in candidates:
        norm = normalize_timestamp(c.get("timestamp"))
        if norm is None:
            continue
        enriched.append((norm["start"], c))

    enriched.sort(key=lambda x: x[0], reverse=reverse)
    return [c for _, c in enriched]


def _strip_ordinal_suffix(text: str) -> str:
    return _ORDINAL_SUFFIX_RE.sub(r"\1", text)


def _clean_time_text(text: str) -> str:
    text = text.strip()
    text = _strip_ordinal_suffix(text)
    text = text.replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _month_str_to_int(mon: str) -> int:
    mon = mon.strip().lower()
    mapping = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    if mon not in mapping:
        raise ValueError(f"can't parse month: {mon}")
    return mapping[mon]

def _parse_single_time_text(text: str) -> Dict[str, Any]:
    """
    - YYYY                    -> [YYYY-01-01, YYYY-12-31], granularity=year
    - YYYY-MM                 -> [YYYY-MM-01, YYYY-MM-last], granularity=month
    - YYYY-MM-DD              -> [same day, same day], granularity=day
    - Apr 2011                -> [2011-04-01, 2011-04-last], granularity=month
    - 27 October 2006         -> [same day, same day], granularity=day
    - January 1, 1981         -> [same day, same day], granularity=day
    - January 1st, 1981       -> [same day, same day], granularity=day
    """
    raw = _clean_time_text(text)

    # YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        y, mo, d = map(int, m.groups())
        dt = date(y, mo, d)
        return {
            "raw": text,
            "start": dt,
            "end": dt,
            "granularity": "day",
            "is_interval": False,
        }

    # YYYY-MM
    m = re.fullmatch(r"(\d{4})-(\d{2})", raw)
    if m:
        y, mo = map(int, m.groups())
        start = date(y, mo, 1)
        end = date(y, mo, _last_day_of_month(y, mo))
        return {
            "raw": text,
            "start": start,
            "end": end,
            "granularity": "month",
            "is_interval": False,
        }

    # YYYY
    m = re.fullmatch(r"(\d{4})", raw)
    if m:
        y = int(m.group(1))
        return {
            "raw": text,
            "start": date(y, 1, 1),
            "end": date(y, 12, 31),
            "granularity": "year",
            "is_interval": False,
        }

    # Apr 2011 / April 2011
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        mon_s, y_s = m.groups()
        mo = _month_str_to_int(mon_s)
        y = int(y_s)
        start = date(y, mo, 1)
        end = date(y, mo, _last_day_of_month(y, mo))
        return {
            "raw": text,
            "start": start,
            "end": end,
            "granularity": "month",
            "is_interval": False,
        }

    # 27 October 2006 / 27 Oct 2006
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        d_s, mon_s, y_s = m.groups()
        d = int(d_s)
        mo = _month_str_to_int(mon_s)
        y = int(y_s)
        dt = date(y, mo, d)
        return {
            "raw": text,
            "start": dt,
            "end": dt,
            "granularity": "day",
            "is_interval": False,
        }

    # January 1 1981 / Jan 1 1981
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})", raw)
    if m:
        mon_s, d_s, y_s = m.groups()
        mo = _month_str_to_int(mon_s)
        d = int(d_s)
        y = int(y_s)
        dt = date(y, mo, d)
        return {
            "raw": text,
            "start": dt,
            "end": dt,
            "granularity": "day",
            "is_interval": False,
        }

    raise ValueError(f"can't parse time expression: {text}")


def normalize_timestamp(
    ts: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    - point: 2015 / 2013-11 / 2015-12-31 / Apr 2011 / 27 October 2006
    - interval: 1992-1998
    - interval: 1977-01-01 to 1983-01-01
    - interval: 1977-01-01 - 1983-01-01
    - interval: 1977-01-01-1983-01-01
    """
    if ts is None:
        return None

    raw = _clean_time_text(str(ts))
    if not raw:
        return None


    m = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})",
        raw,
    )
    if m:
        left, right = m.groups()
        try:
            left_norm = _parse_single_time_text(left)
            right_norm = _parse_single_time_text(right)
            return {
                "raw": ts,
                "start": left_norm["start"],
                "end": right_norm["end"],
                "granularity": None,
                "is_interval": True,
            }
        except Exception:
            return None


    m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", raw)
    if m:
        y1, y2 = map(int, m.groups())
        start = date(y1, 1, 1)
        end = date(y2, 12, 31)
        return {
            "raw": ts,
            "start": start,
            "end": end,
            "granularity": None,
            "is_interval": True,
        }


    m = re.fullmatch(r"(.+?)\s+to\s+(.+)", raw, flags=re.IGNORECASE)
    if m:
        left, right = m.groups()
        left_norm = _parse_single_time_text(left)
        right_norm = _parse_single_time_text(right)
        return {
            "raw": ts,
            "start": left_norm["start"],
            "end": right_norm["end"],
            "granularity": None,
            "is_interval": True,
        }


    if " - " in raw:
        left, right = raw.split(" - ", 1)
        try:
            left_norm = _parse_single_time_text(left)
            right_norm = _parse_single_time_text(right)
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



def normalize_explicit_interval(
    interval_start: str, interval_end: str
) -> Dict[str, Any]:
    start_norm = _parse_single_time_text(interval_start)
    end_norm = _parse_single_time_text(interval_end)
    return {
        "raw": f"{interval_start} -> {interval_end}",
        "start": start_norm["start"],
        "end": end_norm["end"],
        "granularity": None,
        "is_interval": True,
    }


def coarsen_range(
    time_obj: Dict[str, Any],
    granularity: Optional[str],
) -> Dict[str, Any]:

    if granularity is None:
        return time_obj

    start: date = time_obj["start"]
    end: date = time_obj["end"]

    if granularity == "day":

        return {
            **time_obj,
            "start": start,
            "end": end,
        }

    if granularity == "month":
        s = date(start.year, start.month, 1)
        e = date(end.year, end.month, _last_day_of_month(end.year, end.month))
        return {
            **time_obj,
            "start": s,
            "end": e,
        }

    if granularity == "year":
        s = date(start.year, 1, 1)
        e = date(end.year, 12, 31)
        return {
            **time_obj,
            "start": s,
            "end": e,
        }

    raise ValueError(f"不支持的 granularity: {granularity}")





def _ranges_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a["start"] <= b["end"] and a["end"] >= b["start"]


def _range_inside(inner: Dict[str, Any], outer: Dict[str, Any]) -> bool:
    return inner["start"] >= outer["start"] and inner["end"] <= outer["end"]


def _match_without_offset(
    cand_time: Dict[str, Any],
    anchor_time: Dict[str, Any],
    relation: str,
    granularity: Optional[str],
) -> bool:
    if relation == "before":
        return cand_time["end"] < anchor_time["start"]

    if relation == "after":
        return cand_time["start"] > anchor_time["end"]

    if relation == "equal":
        if granularity is None:

            return False
        cand_g = coarsen_range(cand_time, granularity)
        anchor_g = coarsen_range(anchor_time, granularity)
        return _ranges_overlap(cand_g, anchor_g)

    if relation == "inside":
        return _range_inside(cand_time, anchor_time)

    if relation == "overlap":
        return _ranges_overlap(cand_time, anchor_time)

    raise ValueError(f"unknown relation: {relation}")


def _match_with_offset(
    cand_time: Dict[str, Any],
    anchor_time: Dict[str, Any],
    relation: str,
    offset_days: int,
) -> bool:

    if relation == "after":
        targets = [
            anchor_time["start"] + timedelta(days=offset_days),
            anchor_time["end"] + timedelta(days=offset_days),
        ]
    elif relation == "before":
        targets = [
            anchor_time["start"] - timedelta(days=offset_days),
            anchor_time["end"] - timedelta(days=offset_days),
        ]
    else:
        return False

    return any(
        cand_time["start"] <= target <= cand_time["end"]
        for target in targets
    )



def apply_temporal_constraint(
    candidates: List[Dict[str, Any]],
    constraint: Union[Dict[str, Any], Any],
    anchor_candidates: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    use the temporal constraint to filter the candidates for a single question.
    """
    if hasattr(constraint, "model_dump"):
        constraint = constraint.model_dump()

    anchor_kind = constraint["anchor_kind"]
    relation = constraint["relation"]
    granularity = constraint.get("granularity")
    offset_days = constraint.get("offset_days")

    anchor_times: List[Dict[str, Any]] = []

    if anchor_kind == "event":
        if not anchor_candidates:
            return []
        for a in anchor_candidates:
            t = normalize_timestamp(a.get("timestamp"))
            if t is not None:
                anchor_times.append(t)

    elif anchor_kind == "explicit_time":
        t = normalize_timestamp(constraint.get("time_text"))
        if t is not None:
            anchor_times.append(t)

    elif anchor_kind == "explicit_interval":
        start = constraint.get("interval_start")
        end = constraint.get("interval_end")
        if start is not None and end is not None:
            anchor_times.append(normalize_explicit_interval(start, end))

    if not anchor_times:
        return []

    filtered: List[Dict[str, Any]] = []
    seen = set()
    
    # if constraint.get("relation") == "equal" and constraint.get("granularity") is None:
    #     logger.warning(
    #         "[BAD CONSTRAINT ENTER apply_temporal_constraint] constraint=%s",
    #         constraint,
    #     )
    
    for cand in candidates:
        cand_time = normalize_timestamp(cand.get("timestamp"))
        if cand_time is None:
            continue

        matched = False
        for anchor_time in anchor_times:
            if offset_days is not None:
                ok = _match_with_offset(cand_time, anchor_time, relation, offset_days)
            else:
                ok = _match_without_offset(
                    cand_time, anchor_time, relation, granularity
                )

            if ok:
                matched = True
                break

        if matched:
            key = (
                cand.get("head"),
                cand.get("relation"),
                cand.get("tail"),
                cand.get("timestamp"),
            )
            if key not in seen:
                seen.add(key)
                filtered.append(deepcopy(cand))

    return filtered
