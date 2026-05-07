#!/usr/bin/env python3
"""
Compute dataset statistics for LABSHIELD (current schema):
  - open_prp: PRP fields
  - close_vqa: multiple-choice questions
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


VIEWS_DEFAULT = ["head_rgbd", "torso_rgbd", "left_wrist_rgbd", "right_wrist_rgbd"]


def ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
        return [x for x in obj["samples"] if isinstance(x, dict)]
    raise ValueError(f"Unknown dataset format: {type(obj)}")


def parse_action(step: str) -> str:
    s = (step or "").strip()
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", s)
    return m.group(1) if m else ""


def avg(nums: List[int]) -> float:
    return (sum(nums) / len(nums)) if nums else 0.0


def pct(a: int, b: int) -> float:
    return (a / b * 100.0) if b else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/Users/jaspernathaniel/qianpusun/repo/LABSHEILD/data/dataset.json")
    ap.add_argument("--data-dir", default="/Users/jaspernathaniel/qianpusun/repo/LABSHEILD/data")
    ap.add_argument("--views", default=",".join(VIEWS_DEFAULT))
    ap.add_argument("--json-out", default="", help="Optional: write full stats JSON here")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    data_dir = Path(args.data_dir)
    views = [v.strip() for v in (args.views or "").split(",") if v.strip()]

    samples = load_dataset(dataset_path)
    n = len(samples)

    # distributions
    scenario_cnt = Counter()
    op_cnt = Counter()
    safety_cnt = Counter()
    cross = defaultdict(lambda: Counter())  # op -> safety -> count

    # image coverage
    missing_view_path_by_view = Counter()
    missing_file_by_view = Counter()
    total_view_paths = 0

    # label stats
    unsafe_freq = Counter()
    hazard_freq = Counter()
    unsafe_len = []
    hazard_len = []

    # close_vqa stats
    vqa_len = []
    gold_cnt = Counter()
    type_cnt = Counter()
    subtype_cnt = Counter()
    type_sub_cnt = Counter()

    # plan stats
    plan_len = []
    action_freq = Counter()
    lvl2_ok = lvl2_total = 0
    lvl3_ok = lvl3_total = 0

    # alignment
    reasoning_level_mismatch = 0
    decision_missing = 0
    decision_mismatch = 0
    expected_decision = {0: "EXECUTE", 1: "SAFE_SLOW", 2: "STOP_AND_ALERT", 3: "REFUSE"}

    def norm_dec(x: Any) -> str:
        if x is None:
            return ""
        s = str(x).strip().upper().replace("-", "_").replace(" ", "_")
        if s == "EXCUTE":
            s = "EXECUTE"
        return s

    for s in samples:
        scenario = str(s.get("scenario") or "").strip() or "(missing)"
        scenario_cnt[scenario] += 1

        # levels
        op = s.get("operation_level")
        sa = s.get("safety_level")
        try:
            op_i = int(op) if op is not None else None
        except Exception:
            op_i = None
        try:
            sa_i = int(sa) if sa is not None else None
        except Exception:
            sa_i = None

        op_cnt[str(op_i) if op_i is not None else "None"] += 1
        safety_cnt[str(sa_i) if sa_i is not None else "None"] += 1
        if op_i is not None and sa_i is not None:
            cross[str(op_i)][str(sa_i)] += 1

        # images
        obs = ensure_dict(s.get("observation"))
        for v in views:
            rel = None
            if v in obs and isinstance(obs.get(v), dict):
                rel = obs[v].get("path")
            if not rel:
                missing_view_path_by_view[v] += 1
                continue
            total_view_paths += 1
            if not (data_dir / str(rel)).exists():
                missing_file_by_view[v] += 1

        # PRP
        prp = ensure_dict(s.get("open_prp") or s.get("prp"))
        per = ensure_dict(prp.get("perception"))
        rea = ensure_dict(prp.get("reasoning"))
        pla = ensure_dict(prp.get("planning"))

        uf = [str(x).strip() for x in ensure_list(per.get("unsafe_factors")) if str(x).strip()]
        hp = [str(x).strip() for x in ensure_list(rea.get("hazard_patterns")) if str(x).strip()]
        unsafe_len.append(len(uf))
        hazard_len.append(len(hp))
        unsafe_freq.update(uf)
        hazard_freq.update(hp)

        # alignment
        if sa_i is not None:
            rl = rea.get("safety_level", None)
            try:
                rl_i = int(rl) if rl is not None else None
            except Exception:
                rl_i = None
            if rl_i is None or rl_i != sa_i:
                reasoning_level_mismatch += 1

            dec = norm_dec(pla.get("decision", None))
            if not dec:
                decision_missing += 1
            else:
                exp = expected_decision.get(sa_i, "")
                if exp and dec != exp:
                    decision_mismatch += 1

        # close_vqa
        vqa = s.get("close_vqa")
        if vqa is None:
            vqa = s.get("qa_tasks")
        vqa_list = ensure_list(vqa)
        vqa_len.append(len(vqa_list))
        for t in vqa_list:
            if not isinstance(t, dict):
                continue
            gold = str(t.get("gold_choice") or "").strip().upper()
            if gold:
                gold_cnt[gold] += 1
            ty = str(t.get("type") or "").strip()
            st = str(t.get("subtype") or "").strip()
            if ty:
                type_cnt[ty] += 1
            if st:
                subtype_cnt[st] += 1
            if ty and st:
                type_sub_cnt[f"{ty}.{st}"] += 1

        # plan
        steps = [str(x).strip() for x in ensure_list(pla.get("plan_steps")) if str(x).strip()]
        plan_len.append(len(steps))
        for stp in steps:
            a = parse_action(stp)
            if a:
                action_freq[a] += 1

        if sa_i == 2:
            lvl2_total += 1
            try:
                i_stop = next(i for i, x in enumerate(steps) if parse_action(x) == "Stop")
            except StopIteration:
                pass
            else:
                ok = (i_stop + 1 < len(steps)) and (parse_action(steps[i_stop + 1]) == "Alert")
                if ok:
                    lvl2_ok += 1
        if sa_i == 3:
            lvl3_total += 1
            if [parse_action(x) for x in steps] == ["Refuse", "Alert"]:
                lvl3_ok += 1

    summary: Dict[str, Any] = {
        "num_samples": n,
        "scenario_distribution": dict(scenario_cnt),
        "operation_level_distribution": dict(op_cnt),
        "safety_level_distribution": dict(safety_cnt),
        "op_safety_crosstab": {op: dict(cross[op]) for op in sorted(cross.keys())},
        "image": {
            "views": views,
            "total_view_paths": total_view_paths,
            "missing_view_path_by_view": dict(missing_view_path_by_view),
            "missing_file_by_view": dict(missing_file_by_view),
        },
        "labels": {
            "unsafe_factors_unique": len(unsafe_freq),
            "hazard_patterns_unique": len(hazard_freq),
            "unsafe_factors_avg_per_sample": avg(unsafe_len),
            "hazard_patterns_avg_per_sample": avg(hazard_len),
            "unsafe_factors_top10": unsafe_freq.most_common(10),
            "hazard_patterns_top10": hazard_freq.most_common(10),
        },
        "close_vqa": {
            "tasks_total": sum(vqa_len),
            "tasks_per_sample_min": min(vqa_len) if vqa_len else 0,
            "tasks_per_sample_max": max(vqa_len) if vqa_len else 0,
            "tasks_per_sample_avg": avg(vqa_len),
            "gold_choice_distribution": dict(gold_cnt),
            "type_top10": type_cnt.most_common(10),
            "subtype_top10": subtype_cnt.most_common(10),
            "type_subtype_top10": type_sub_cnt.most_common(10),
        },
        "planning": {
            "plan_steps_total": sum(plan_len),
            "plan_steps_per_sample_min": min(plan_len) if plan_len else 0,
            "plan_steps_per_sample_max": max(plan_len) if plan_len else 0,
            "plan_steps_per_sample_avg": avg(plan_len),
            "action_top15": action_freq.most_common(15),
            "level2_stop_then_alert_ok": {"ok": lvl2_ok, "total": lvl2_total, "rate_pct": pct(lvl2_ok, lvl2_total)},
            "level3_refuse_and_alert_ok": {"ok": lvl3_ok, "total": lvl3_total, "rate_pct": pct(lvl3_ok, lvl3_total)},
        },
        "alignment": {
            "reasoning_safety_level_mismatch_samples": reasoning_level_mismatch,
            "decision_missing_samples": decision_missing,
            "decision_mismatch_samples": decision_mismatch,
        },
    }

    # print numbers
    print("num_samples=", summary["num_samples"])
    print("scenario_distribution=", summary["scenario_distribution"])
    print("operation_level_distribution=", summary["operation_level_distribution"])
    print("safety_level_distribution=", summary["safety_level_distribution"])
    print("op_safety_crosstab=", summary["op_safety_crosstab"])
    print("image_missing_view_path_by_view=", summary["image"]["missing_view_path_by_view"])
    print("image_missing_file_by_view=", summary["image"]["missing_file_by_view"])
    print("labels_unsafe_unique=", summary["labels"]["unsafe_factors_unique"], "avg_per_sample=", f"{summary['labels']['unsafe_factors_avg_per_sample']:.2f}")
    print("labels_hazard_unique=", summary["labels"]["hazard_patterns_unique"], "avg_per_sample=", f"{summary['labels']['hazard_patterns_avg_per_sample']:.2f}")
    print("labels_unsafe_top10=", summary["labels"]["unsafe_factors_top10"])
    print("labels_hazard_top10=", summary["labels"]["hazard_patterns_top10"])
    print(
        "close_vqa_tasks_total=",
        summary["close_vqa"]["tasks_total"],
        "per_sample(min/avg/max)=",
        summary["close_vqa"]["tasks_per_sample_min"],
        f"{summary['close_vqa']['tasks_per_sample_avg']:.2f}",
        summary["close_vqa"]["tasks_per_sample_max"],
    )
    print("close_vqa_gold_choice_distribution=", summary["close_vqa"]["gold_choice_distribution"])
    print("close_vqa_type_top10=", summary["close_vqa"]["type_top10"])
    print("close_vqa_subtype_top10=", summary["close_vqa"]["subtype_top10"])
    print(
        "planning_plan_steps_total=",
        summary["planning"]["plan_steps_total"],
        "per_sample(min/avg/max)=",
        summary["planning"]["plan_steps_per_sample_min"],
        f"{summary['planning']['plan_steps_per_sample_avg']:.2f}",
        summary["planning"]["plan_steps_per_sample_max"],
    )
    print("planning_action_top15=", summary["planning"]["action_top15"])
    print("level2_stop_then_alert_ok=", summary["planning"]["level2_stop_then_alert_ok"])
    print("level3_refuse_and_alert_ok=", summary["planning"]["level3_refuse_and_alert_ok"])
    print("alignment=", summary["alignment"])

    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("saved_json=", str(outp))


if __name__ == "__main__":
    main()


