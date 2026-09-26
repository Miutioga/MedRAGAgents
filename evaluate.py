"""
evaluate.py — Evaluation Module for MedRAGAgents
──────────────────────────────────────────────────
Computes all metrics required by the midterm spec:

  Primary:
    • Accuracy = Correct / Total
    • Invalid response rate

  Secondary:
    • Accuracy gain  = Accuracy(V3) − Accuracy(V0)
    • Per-variant leaderboard table
    • Paired Win/Loss/Tie comparison
    • Simple error analysis
    • Bootstrap 95% CI (optional)
    • McNemar test (optional)

Usage (CLI):
    python evaluate.py --pred_dir ./outputs --dataset_dir ./datasets/MedQA/

Usage (Python):
    from evaluate import Evaluator
    ev = Evaluator(pred_dir="./outputs", dataset_dir="./datasets/MedQA/")
    ev.print_leaderboard()
"""

import os
import re
import json
import math
import random
import argparse
import jsonlines
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _load_predictions(fpath: str) -> List[dict]:
    records = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _is_correct(pred: str, gold: str) -> bool:
    if not pred or pred == "ERROR.":
        return False
    return pred.strip().upper() == gold.strip().upper()


def _is_invalid(pred: str) -> bool:
    return not pred or pred == "" or pred == "ERROR."


# ─────────────────────────────────────────────────────────────
# Core Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(records: List[dict]) -> dict:
    """Compute accuracy and invalid rate for a list of prediction records."""
    total    = len(records)
    correct  = 0
    invalid  = 0
    for r in records:
        pred = r.get("pred_answer", "")
        gold = r.get("gold_answer", "")
        if _is_invalid(pred):
            invalid += 1
        elif _is_correct(pred, gold):
            correct += 1

    accuracy      = correct / total if total else 0.0
    invalid_rate  = invalid / total if total else 0.0
    return {
        "total":        total,
        "correct":      correct,
        "invalid":      invalid,
        "accuracy":     accuracy,
        "invalid_rate": invalid_rate,
    }


# ─────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────

def bootstrap_ci(records: List[dict], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Return (lower, upper) 95% CI for accuracy via bootstrap."""
    corrects = [int(_is_correct(r.get("pred_answer",""), r.get("gold_answer","")))
                for r in records]
    n = len(corrects)
    boot_means = []
    for _ in range(n_boot):
        sample = random.choices(corrects, k=n)
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(alpha/2 * n_boot)]
    hi = boot_means[int((1 - alpha/2) * n_boot)]
    return lo, hi


# ─────────────────────────────────────────────────────────────
# McNemar Test
# ─────────────────────────────────────────────────────────────

def mcnemar_test(records_a: List[dict], records_b: List[dict]) -> dict:
    """
    McNemar test between two systems on paired samples.
    Returns {'b': int, 'c': int, 'statistic': float, 'p_approx': float}
    """
    # align by question
    qa = {r["question"]: r for r in records_a}
    qb = {r["question"]: r for r in records_b}
    common = set(qa.keys()) & set(qb.keys())

    b, c = 0, 0   # A correct B wrong | A wrong B correct
    for q in common:
        a_ok = _is_correct(qa[q].get("pred_answer",""), qa[q].get("gold_answer",""))
        b_ok = _is_correct(qb[q].get("pred_answer",""), qb[q].get("gold_answer",""))
        if a_ok and not b_ok:
            b += 1
        elif not a_ok and b_ok:
            c += 1

    # Continuity-corrected McNemar statistic
    if b + c == 0:
        stat, p = 0.0, 1.0
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        # χ²(1) approximation: p ≈ 2*(1 - Φ(√stat)) — simplified
        from math import erfc, sqrt
        p = erfc(sqrt(stat / 2))
    return {"b": b, "c": c, "statistic": round(stat, 4), "p_approx": round(p, 4)}


# ─────────────────────────────────────────────────────────────
# Win / Loss / Tie
# ─────────────────────────────────────────────────────────────

def win_loss_tie(records_base: List[dict], records_full: List[dict]) -> dict:
    qa = {r["question"]: r for r in records_base}
    qb = {r["question"]: r for r in records_full}
    common = set(qa.keys()) & set(qb.keys())

    win, loss, tie = 0, 0, 0
    for q in common:
        a_ok = _is_correct(qa[q].get("pred_answer",""), qa[q].get("gold_answer",""))
        b_ok = _is_correct(qb[q].get("pred_answer",""), qb[q].get("gold_answer",""))
        if b_ok and not a_ok:
            win += 1
        elif a_ok and not b_ok:
            loss += 1
        else:
            tie += 1
    total = len(common)
    return {
        "win":  win,
        "loss": loss,
        "tie":  tie,
        "total": total,
        "win_rate":  round(win/total, 4) if total else 0,
        "loss_rate": round(loss/total, 4) if total else 0,
    }


# ─────────────────────────────────────────────────────────────
# Error Analysis
# ─────────────────────────────────────────────────────────────

def error_analysis(records: List[dict], top_n: int = 10) -> List[dict]:
    """Return up to top_n wrong predictions for inspection."""
    errors = []
    for r in records:
        pred = r.get("pred_answer", "")
        gold = r.get("gold_answer", "")
        if not _is_correct(pred, gold):
            errors.append({
                "question":    r.get("question", "")[:120] + "…",
                "gold":        gold,
                "pred":        pred,
                "is_invalid":  _is_invalid(pred),
            })
    return errors[:top_n]


# ─────────────────────────────────────────────────────────────
# Evaluator (main class)
# ─────────────────────────────────────────────────────────────

class Evaluator:
    """
    Load all prediction files from `pred_dir` (one per variant)
    and compute the full evaluation suite.

    Prediction file naming convention:
        outputs/<variant>_predictions.jsonl
        e.g. outputs/V0_predictions.jsonl
    """

    VARIANTS = ["V0", "V1", "V2", "V3"]

    def __init__(self, pred_dir: str = "./outputs", dataset_dir: str = None):
        self.pred_dir    = pred_dir
        self.dataset_dir = dataset_dir
        self.data: Dict[str, List[dict]] = {}
        self._load_all()

    def _load_all(self):
        for variant in self.VARIANTS:
            fpath = os.path.join(self.pred_dir, f"{variant}_predictions.jsonl")
            if os.path.exists(fpath):
                self.data[variant] = _load_predictions(fpath)
                print(f"[Evaluator] Loaded {len(self.data[variant])} records for {variant}")
            else:
                print(f"[Evaluator] No predictions found for {variant} at {fpath}")

    # ── leaderboard ──────────────────────────────────────────
    def leaderboard(self) -> List[dict]:
        rows = []
        for variant in self.VARIANTS:
            if variant not in self.data:
                continue
            m = compute_metrics(self.data[variant])
            ci_lo, ci_hi = bootstrap_ci(self.data[variant])
            rows.append({
                "Variant":       variant,
                "Total":         m["total"],
                "Correct":       m["correct"],
                "Accuracy":      f"{m['accuracy']:.4f}",
                "Invalid Rate":  f"{m['invalid_rate']:.4f}",
                "95% CI":        f"({ci_lo:.4f}, {ci_hi:.4f})",
            })
        return rows

    # ── paired comparison ────────────────────────────────────
    def paired_comparison(self, base: str = "V0", full: str = "V3") -> dict:
        if base not in self.data or full not in self.data:
            return {}
        wlt = win_loss_tie(self.data[base], self.data[full])
        mc  = mcnemar_test(self.data[base], self.data[full])
        return {"win_loss_tie": wlt, "mcnemar": mc}

    # ── report generation ─────────────────────────────────────
    def generate_text_report(self) -> str:
        lines = []
        rows = self.leaderboard()
        if not rows:
            return "No prediction files found."

        header = ["Variant", "Total", "Correct", "Accuracy", "Invalid Rate", "95% CI"]
        col_w = [max(len(h), max(len(str(r[h])) for r in rows)) + 2 for h in header]
        sep   = "+" + "+".join("-" * w for w in col_w) + "+"
        fmt   = "|" + "|".join(f" {{:<{w-1}}}" for w in col_w) + "|"

        lines.append("=" * 70)
        lines.append("  MedRAGAgents BENCHMARK LEADERBOARD (PLAIN TEXT TABLE)")
        lines.append("=" * 70)
        lines.append(sep)
        lines.append(fmt.format(*header))
        lines.append(sep)
        for r in rows:
            lines.append(fmt.format(*[str(r[h]) for h in header]))
        lines.append(sep)

        if "V0" in self.data and "V3" in self.data:
            v0_acc = compute_metrics(self.data["V0"])["accuracy"]
            v3_acc = compute_metrics(self.data["V3"])["accuracy"]
            lines.append(f"\n  Accuracy Gain (V3 - V0): {v3_acc - v0_acc:+.4f}\n")

        r_paired = self.paired_comparison("V0", "V3")
        if r_paired:
            wlt = r_paired["win_loss_tie"]
            mc = r_paired["mcnemar"]
            lines.append("-" * 50)
            lines.append("  Paired Comparison: V0 (baseline) vs V3 (full)")
            lines.append("-" * 50)
            lines.append(f"  Win  (V3 correct, V0 wrong): {wlt['win']} ({wlt['win_rate']*100:.1f}%)")
            lines.append(f"  Loss (V0 correct, V3 wrong): {wlt['loss']} ({wlt['loss_rate']*100:.1f}%)")
            lines.append(f"  Tie  (Both correct / wrong): {wlt['tie']}")
            lines.append(f"  McNemar statistic           : {mc['statistic']} (p ≈ {mc['p_approx']})\n")

        if "V3" in self.data:
            errors = error_analysis(self.data["V3"], top_n=10)
            lines.append("-" * 50)
            lines.append("  Error Analysis — V3 (top 10)")
            lines.append("-" * 50)
            for i, e in enumerate(errors, 1):
                tag = "[INVALID]" if e["is_invalid"] else f"[WRONG: pred={e['pred']}, gold={e['gold']}]"
                lines.append(f"  {i:02d}. {tag}")
                lines.append(f"      Q: {e['question']}")
            lines.append("")

        return "\n".join(lines)

    def print_leaderboard(self):
        print(self.generate_text_report())

    def print_paired_comparison(self, base="V0", full="V3"):
        r = self.paired_comparison(base, full)
        if not r:
            print(f"Cannot compare {base} vs {full}: missing data.")
            return
        wlt = r["win_loss_tie"]
        mc  = r["mcnemar"]
        print(f"\n{'─'*50}")
        print(f"  Paired Comparison: {base} (baseline) vs {full} (full)")
        print(f"{'─'*50}")
        print(f"  Win  (V3 correct, V0 wrong): {wlt['win']}")
        print(f"  Loss (V0 correct, V3 wrong): {wlt['loss']}")
        print(f"  Tie :                         {wlt['tie']}")
        print(f"  Win rate: {wlt['win_rate']:.4f} | Loss rate: {wlt['loss_rate']:.4f}")
        print(f"  McNemar χ²={mc['statistic']}, p≈{mc['p_approx']}")
        print()

    def print_error_analysis(self, variant: str = "V3", top_n: int = 10):
        if variant not in self.data:
            print(f"No data for {variant}")
            return
        errors = error_analysis(self.data[variant], top_n=top_n)
        print(f"\n{'─'*50}")
        print(f"  Error Analysis — {variant} (top {top_n})")
        print(f"{'─'*50}")
        for i, e in enumerate(errors, 1):
            tag = "[INVALID]" if e["is_invalid"] else f"[WRONG: pred={e['pred']}, gold={e['gold']}]"
            print(f"  {i:02d}. {tag}")
            print(f"      Q: {e['question']}")
        print()

    def save_reports(self, output_dir: str = None):
        target_dir = output_dir or self.pred_dir
        os.makedirs(target_dir, exist_ok=True)

        txt_report = self.generate_text_report()
        txt_path = os.path.join(target_dir, "evaluation_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_report)
        print(f"[Evaluator] Saved Plain Text Report to {txt_path}")

    # ── full report ──────────────────────────────────────────
    def full_report(self):
        txt_report = self.generate_text_report()
        print("\n" + txt_report)
        self.save_reports()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedRAGAgents Evaluation")
    parser.add_argument("--pred_dir",    default="./outputs",         help="Directory with *_predictions.jsonl")
    parser.add_argument("--dataset_dir", default="./datasets/MedQA/", help="Dataset directory (optional)")
    parser.add_argument("--base",        default="V0", help="Baseline variant for paired comparison")
    parser.add_argument("--full",        default="V3", help="Full variant for paired comparison")
    args = parser.parse_args()

    ev = Evaluator(pred_dir=args.pred_dir, dataset_dir=args.dataset_dir)
    ev.full_report()


