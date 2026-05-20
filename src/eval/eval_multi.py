from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Union
import json
import os
from datetime import datetime

from src.utils.common_utils import FileHelper


def _norm(x: str) -> str:
    return (x or "").strip().lower()


def _as_list(x: Union[str, Iterable[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return []


def _hit_at_k(preds: List[str], golds: List[str], k: int) -> int:
    if not preds or not golds:
        return 0
    preds_k = preds[:k]
    gold_set = set(golds)
    return int(any(p in gold_set for p in preds_k))


def compute_hits(
    records: List[Dict[str, Any]],
    ks: Tuple[int, ...] = (1, 10),
    skip_if_no_gold: bool = True,
) -> Dict[str, Any]:

    overall_cnt = 0
    overall_hits = {k: 0 for k in ks}

    by_qlabel_cnt = defaultdict(int)
    by_qlabel_hits = defaultdict(lambda: {k: 0 for k in ks})

    by_type_cnt = defaultdict(int)
    by_type_hits = defaultdict(lambda: {k: 0 for k in ks})

    for r in records:
        if not r:
            continue

        gold_raw = r.get("answers") or r.get("answer")
        gold_list = [_norm(s) for s in _as_list(gold_raw) if _norm(s)]
        if skip_if_no_gold and not gold_list:
            continue

        preds_raw = r.get("predict_answer")
        preds_list = [_norm(s) for s in _as_list(preds_raw) if _norm(s)]

        atype_raw = (r.get("answer_type") or "").strip()
        atype_raw_lc = atype_raw.lower()

        qlabel = (r.get("qlabel") or "").strip()
        if qlabel not in ("Single", "Multiple"):
            qlabel = "_OTHER_"

        if atype_raw_lc == "time":
            atype = "Time"
        elif atype_raw_lc == "entity":
            atype = "Entity"
        else:
            atype = "_OTHER_"

        overall_cnt += 1
        by_qlabel_cnt[qlabel] += 1
        by_type_cnt[atype] += 1

        for k in ks:
            hit = _hit_at_k(preds_list, gold_list, k)

            overall_hits[k] += hit
            by_qlabel_hits[qlabel][k] += hit
            by_type_hits[atype][k] += hit

    def q_cnt(label: str) -> int:
        return by_qlabel_cnt.get(label, 0)

    def t_cnt(label: str) -> int:
        return by_type_cnt.get(label, 0)

    def q_hit(label: str, k: int) -> int:
        return by_qlabel_hits[label][k] if label in by_qlabel_hits else 0

    def t_hit(label: str, k: int) -> int:
        return by_type_hits[label][k] if label in by_type_hits else 0

    def safe_rate(h: int, c: int) -> float:
        return h / c if c else 0.0

    # -------- Hit@1 --------
    k1 = 1
    hit1_overall = safe_rate(overall_hits.get(k1, 0), overall_cnt)
    hit1_Multiple_hits = q_hit("Multiple", k1)
    hit1_Single_hits = q_hit("Single", k1)
    hit1_Entity_hits = t_hit("Entity", k1)
    hit1_Time_hits = t_hit("Time", k1)

    hit1 = {
        "overall": hit1_overall,
        "Multiple": safe_rate(hit1_Multiple_hits, q_cnt("Multiple")),
        "Single": safe_rate(hit1_Single_hits, q_cnt("Single")),
        "Entity": safe_rate(hit1_Entity_hits, t_cnt("Entity")),
        "Time": safe_rate(hit1_Time_hits, t_cnt("Time")),
    }

    # -------- Hit@10 --------
    k10 = 10
    hit10_overall = safe_rate(overall_hits.get(k10, 0), overall_cnt)
    hit10_Multiple_hits = q_hit("Multiple", k10)
    hit10_Single_hits = q_hit("Single", k10)
    hit10_Entity_hits = t_hit("Entity", k10)
    hit10_Time_hits = t_hit("Time", k10)

    hit10 = {
        "overall": hit10_overall,
        "Multiple": safe_rate(hit10_Multiple_hits, q_cnt("Multiple")),
        "Single": safe_rate(hit10_Single_hits, q_cnt("Single")),
        "Entity": safe_rate(hit10_Entity_hits, t_cnt("Entity")),
        "Time": safe_rate(hit10_Time_hits, t_cnt("Time")),
    }

    # -------- count --------
    count = {
        "all": overall_cnt,
        "Multiple": q_cnt("Multiple"),
        "Single": q_cnt("Single"),
        "Entity": t_cnt("Entity"),
        "Time": t_cnt("Time"),
    }

    # -------- hit_count--------
    hit_count = {
        "Hit@1": {
            "all": overall_hits.get(k1, 0),
            "Single": hit1_Single_hits,
            "Multiple": hit1_Multiple_hits,
            "Entity": hit1_Entity_hits,
            "Time": hit1_Time_hits,
        },
        "Hit@10": {
            "all": overall_hits.get(k10, 0),
            "Single": hit10_Single_hits,
            "Multiple": hit10_Multiple_hits,
            "Entity": hit10_Entity_hits,
            "Time": hit10_Time_hits,
        },
    }

    result = {
        "Hit@1": hit1,
        "Hit@10": hit10,
        "count": count,
        "hit_count": hit_count,
        "timestamp": datetime.now().isoformat(),
    }

    return result


def print_hit_table_pretty(report: dict):

    levels = ["overall", "Multiple", "Single", "Entity", "Time"]
    col_width = 10

    hit1 = report.get("Hit@1", {})
    hit10 = report.get("Hit@10", {})

    sep_top = "┌────────────" + "┬" + "┬".join(["─" * col_width for _ in levels]) + "┐"
    sep_mid = "├────────────" + "┼" + "┼".join(["─" * col_width for _ in levels]) + "┤"
    sep_bot = "└────────────" + "┴" + "┴".join(["─" * col_width for _ in levels]) + "┘"

    header = "│ Metric     │" + "".join(f"{lvl:^{col_width}}│" for lvl in levels)

    print(sep_top)
    print(header)
    print(sep_mid)

    row1 = "│ Hit@1      │" + "".join(
        f"{float(hit1.get(lvl, 0.0)):^{col_width}.3f}│" for lvl in levels
    )
    print(row1)
    print(sep_mid)

    row2 = "│ Hit@10     │" + "".join(
        f"{float(hit10.get(lvl, 0.0)):^{col_width}.3f}│" for lvl in levels
    )
    print(row2)
    print(sep_bot)


def final_eval(source_file):
    data = FileHelper.load_json(source_file)
    report = compute_hits(data)

    print_hit_table_pretty(report)

    run_dir = os.path.dirname(source_file)
    base_dir = os.path.dirname(run_dir)
    eval_log_path = os.path.join(base_dir, "eval_log.jsonl")

    log_record = dict(report)

    with open(eval_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    return report
