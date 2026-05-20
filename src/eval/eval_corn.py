import json
import re, os
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Union


from src.utils.common_utils import FileHelper
from src.config.base_config import logger


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
    skip_if_no_gold: bool = False,
) -> Dict[str, Any]:

    overall_cnt = 0
    overall_hits = {k: 0 for k in ks}

    by_level_cnt = defaultdict(int)
    by_level_hits = defaultdict(lambda: {k: 0 for k in ks})

    for r in records:
        gold_raw = r.get("answer")
        pred_raw = r.get("predict_answer")

        gold_list = [_norm(s) for s in _as_list(gold_raw) if _norm(s)]
        preds_list = [_norm(s) for s in _as_list(pred_raw) if _norm(s)]

        if skip_if_no_gold and not gold_list:
            continue

        qlevel = (r.get("question_level") or "_OTHER_").strip()

        overall_cnt += 1
        by_level_cnt[qlevel] += 1

        for k in ks:
            hit = _hit_at_k(preds_list, gold_list, k)
            overall_hits[k] += hit
            by_level_hits[qlevel][k] += hit

    def build_hit_struct(k: int):
        return {
            "overall": overall_hits[k] / overall_cnt if overall_cnt else 0.0,
            "simple": (
                by_level_hits["simple"][k] / by_level_cnt["simple"]
                if by_level_cnt["simple"]
                else 0.0
            ),
            "medium": (
                by_level_hits["medium"][k] / by_level_cnt["medium"]
                if by_level_cnt["medium"]
                else 0.0
            ),
            "complex": (
                by_level_hits["complex"][k] / by_level_cnt["complex"]
                if by_level_cnt["complex"]
                else 0.0
            ),
        }

    def build_hit_count(k: int):
        return {
            "all": overall_hits[k],
            "simple": by_level_hits["simple"][k],
            "medium": by_level_hits["medium"][k],
            "complex": by_level_hits["complex"][k],
        }

    result = {
        "Hit@1": build_hit_struct(1),
        "Hit@10": build_hit_struct(10),
        "count": {
            "all": overall_cnt,
            "simple": by_level_cnt["simple"],
            "medium": by_level_cnt["medium"],
            "complex": by_level_cnt["complex"],
        },
        "hit_count": {"Hit@1": build_hit_count(1), "Hit@10": build_hit_count(10)},
        "timestamp": datetime.now().isoformat(),
    }

    return result


def print_hit_table_pretty(report: dict):

    hit1 = report.get("Hit@1", {})
    hit10 = report.get("Hit@10", {})

    level_order = ["overall", "simple", "medium", "complex"]
    col_width = 10

    sep_top = (
        "┌────────────" + "┬" + "┬".join(["─" * col_width for _ in level_order]) + "┐"
    )
    sep_mid = (
        "├────────────" + "┼" + "┼".join(["─" * col_width for _ in level_order]) + "┤"
    )
    sep_bot = (
        "└────────────" + "┴" + "┴".join(["─" * col_width for _ in level_order]) + "┘"
    )

    header = "│ Metric     │" + "".join(f"{lvl:^{col_width}}│" for lvl in level_order)

    print(sep_top)
    print(header)
    print(sep_mid)

    # Hits@1 row
    row1 = "│ Hits@1     │" + "".join(
        f"{hit1.get(lvl, 0.0):^{col_width}.3f}│" for lvl in level_order
    )
    print(row1)
    print(sep_mid)

    # Hits@10 row
    row2 = "│ Hits@10    │" + "".join(
        f"{hit10.get(lvl, 0.0):^{col_width}.3f}│" for lvl in level_order
    )
    print(row2)
    print(sep_bot)


def append_jsonl(path: str, obj: dict):

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def final_eval(source_file):

    data = FileHelper.load_json(source_file)

    report = compute_hits(data)

    print_hit_table_pretty(report)

    run_dir = os.path.dirname(source_file)
    base_dir = os.path.dirname(run_dir)
    jsonl_log_file = os.path.join(base_dir, "eval_log.jsonl")

    if jsonl_log_file:
        log_record = dict(report)
        append_jsonl(jsonl_log_file, log_record)
    return report
