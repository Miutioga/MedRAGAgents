"""
run.py — CLI Entry Point for MedRAGAgents
──────────────────────────────────────────
Runs a specified variant (V0–V3) on MedQA-USMLE data
and writes predictions to outputs/<VARIANT>_predictions.jsonl.

Examples:
    # Quick smoke-test with 10 questions
    python run.py --variant V0 --n 10

    # Full V3 run with Gemini
    python run.py --variant V3 --llm google/gemini-1.5-flash

    # Run all variants sequentially
    python run.py --variant ALL --n 100

    # Evaluate after running
    python evaluate.py --pred_dir ./outputs
"""

import os
import sys
import json
import argparse
import tqdm
import jsonlines

sys.path.insert(0, os.path.dirname(__file__))

from pipeline  import build_pipeline, VARIANT_MAP
from evaluate  import Evaluator
import config as cfg


# ─────────────────────────────────────────────────────────────
# Dataset loader (MedQA JSONL format)
# ─────────────────────────────────────────────────────────────

def load_dataset(dataset_dir: str, split: str = "test") -> list:
    fpath = os.path.join(dataset_dir, f"{split}.jsonl")
    if not os.path.exists(fpath):
        # Try .json
        fpath = os.path.join(dataset_dir, f"{split}.json")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Dataset not found: {fpath}")
    data = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def extract_fields(item: dict, dataset_name: str = "MedQA"):
    """Normalise a dataset record to (question, options, gold_answer)."""
    question    = item.get("question", "")
    options     = item.get("options", {})
    gold_answer = item.get("answer_idx", item.get("answer", ""))
    return question, options, gold_answer


# ─────────────────────────────────────────────────────────────
# Single variant runner
# ─────────────────────────────────────────────────────────────

def run_variant(
    variant:      str,
    data:         list,
    llm_name:     str,
    output_dir:   str,
    start:        int = 0,
    end:          int = -1,
    dataset_name: str = "MedQA",
    delay:        float = 2.0,
):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{variant}_predictions.jsonl")

    pipeline = build_pipeline(variant, llm_name=llm_name)

    end_pos  = len(data) if end == -1 else min(end, len(data))
    test_ids = range(start, end_pos)

    print(f"\n{'='*55}")
    print(f"  Running {variant} | {llm_name} | questions {start}–{end_pos-1}")
    print(f"  Output → {out_path}")
    print(f"{'='*55}")

    results = []
    for idx in tqdm.tqdm(test_ids, desc=f"[{variant}]"):
        item = data[idx]
        question, options, gold = extract_fields(item, dataset_name)

        try:
            record = pipeline.run(question=question, options=options, gold_answer=gold)
        except Exception as e:
            print(f"\n  [ERROR] idx={idx}: {e}")
            record = {
                "question":    question,
                "options":     options,
                "gold_answer": gold,
                "pred_answer": "",
                "raw_output":  f"ERROR: {e}",
            }

        record["idx"] = idx
        results.append(record)

        # Append to file incrementally (safe against crashes)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if delay > 0 and idx != test_ids[-1]:
            import time
            time.sleep(delay)

    # Summary
    correct = sum(
        1 for r in results
        if r.get("pred_answer", "").upper() == r.get("gold_answer", "").upper()
           and r.get("pred_answer", "") not in ("", "ERROR.")
    )
    total = len(results)
    print(f"\n  [{variant}] Accuracy: {correct}/{total} = {correct/total:.4f}")
    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedRAGAgents Runner")
    parser.add_argument("--variant",      default="V3",
                        choices=list(VARIANT_MAP.keys()) + ["ALL"],
                        help="Pipeline variant to run (V0/V1/V2/V3/ALL)")
    parser.add_argument("--llm",          default=cfg.DEFAULT_LLM,
                        help="LLM name, e.g. google/gemini-1.5-flash")
    parser.add_argument("--dataset_dir",  default=cfg.DATASET_DIR,
                        help="Path to MedQA dataset directory")
    parser.add_argument("--dataset_name", default="MedQA",
                        help="Dataset name (MedQA / MedMCQA / PubMedQA / MMLU)")
    parser.add_argument("--split",        default="test",
                        help="Dataset split: train / dev / test")
    parser.add_argument("--start",        type=int, default=0,
                        help="Start index (inclusive)")
    parser.add_argument("--n",            type=int, default=-1,
                        help="Number of questions to run (-1 = all)")
    parser.add_argument("--output_dir",   default=cfg.OUTPUT_DIR,
                        help="Output directory for prediction files")
    parser.add_argument("--delay",        type=float, default=2.0,
                        help="Delay in seconds between questions to respect API rate limits")
    parser.add_argument("--evaluate",     action="store_true",
                        help="Run evaluation after inference")
    args = parser.parse_args()

    # Compute end index
    end_pos = args.start + args.n if args.n > 0 else -1

    # Load data
    print(f"[run.py] Loading {args.split} data from {args.dataset_dir} …")
    try:
        data = load_dataset(args.dataset_dir, split=args.split)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"[run.py] Loaded {len(data)} records.")

    # Run variants
    variants = list(VARIANT_MAP.keys()) if args.variant == "ALL" else [args.variant]
    for v in variants:
        run_variant(
            variant=v,
            data=data,
            llm_name=args.llm,
            output_dir=args.output_dir,
            start=args.start,
            end=end_pos,
            dataset_name=args.dataset_name,
            delay=args.delay,
        )

    # Evaluate
    if args.evaluate:
        print("\n" + "="*55)
        print("  EVALUATION RESULTS")
        print("="*55)
        ev = Evaluator(pred_dir=args.output_dir)
        ev.full_report()


if __name__ == "__main__":
    main()
