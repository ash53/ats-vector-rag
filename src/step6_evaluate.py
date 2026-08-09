"""
STEP 6 — Evaluation

Runs the full pipeline on a stratified sample of held-out cases and
measures decision quality against ground truth labels.

Metrics reported: Accuracy, Precision, Recall, F1 (SELECT class),
and a confusion matrix.

Results are saved to data/processed/eval_results.json.

Usage:
    python src/step6_evaluate.py              # evaluate 20 cases (default)
    python src/step6_evaluate.py --n 50       # evaluate 50 cases
    python src/step6_evaluate.py --n 10 --seed 99
"""

# -------------------------------
# macOS / PyTorch stability fixes
# -------------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -------------------------------
# Imports
# -------------------------------
import json
import time
import random
import argparse
from pathlib import Path

# Import the full pipeline from step5
from step5_llm_decision import decide

# -------------------------------
# File paths
# -------------------------------
CASES_PATH   = Path("data/processed/cases_stage2.jsonl")
RESULTS_PATH = Path("data/processed/eval_results.json")

# -------------------------------
# Load all cases
# -------------------------------
def load_cases(path: Path) -> list:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))
    return cases

# -------------------------------
# Stratified sample
# Ensures ~equal select/reject in the test set
# -------------------------------
def stratified_sample(cases: list, n: int, seed: int) -> list:
    rng = random.Random(seed)

    select_cases = [c for c in cases if c["decision"].lower() == "select"]
    reject_cases = [c for c in cases if c["decision"].lower() == "reject"]

    n_each = n // 2
    sampled = (
        rng.sample(select_cases, min(n_each, len(select_cases))) +
        rng.sample(reject_cases, min(n_each, len(reject_cases)))
    )
    rng.shuffle(sampled)
    return sampled

# -------------------------------
# Metrics
# -------------------------------
def compute_metrics(results: list) -> dict:
    tp = fp = fn = tn = 0

    for r in results:
        pred  = r["prediction"].lower()
        truth = r["ground_truth"].lower()

        if pred == "select" and truth == "select":
            tp += 1
        elif pred == "select" and truth != "select":
            fp += 1
        elif pred != "select" and truth == "select":
            fn += 1
        else:
            tn += 1

    total    = tp + fp + fn + tn
    accuracy  = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp)   if (tp + fp) else 0.0
    recall    = tp / (tp + fn)   if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    # Share of cases the system called SELECT. Both LLM hiring pipelines in
    # this project over-predict SELECT, so this is worth reporting next to
    # accuracy: a system that answers "select" every time scores 50% on a
    # balanced test set without doing anything.
    pred_select_rate = (tp + fp) / total if total else 0.0

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "pred_select_rate": round(pred_select_rate, 4),
        "confusion_matrix": {
            "TP (pred select, actual select)": tp,
            "FP (pred select, actual reject)": fp,
            "FN (pred reject, actual select)": fn,
            "TN (pred reject, actual reject)": tn,
        },
        "total_cases": total,
    }

# -------------------------------
# Main
# -------------------------------
def main(n: int = 20, seed: int = 42, out: Path = RESULTS_PATH):
    assert CASES_PATH.exists(), "cases_stage2.jsonl not found. Run steps 1–2 first."

    all_cases = load_cases(CASES_PATH)
    test_cases = stratified_sample(all_cases, n, seed)

    print(f"Evaluating {len(test_cases)} cases (seed={seed})...\n")

    results = []

    for i, case in enumerate(test_cases, 1):
        case_id    = case["case_id"]
        role       = case["role"]
        cv_text    = case["cv_text"]
        jd_text    = case["jd_text"]
        skills_cv  = case["entities"]["skills_cv"]
        skills_jd  = case["entities"]["skills_jd"]
        truth      = case["decision"].lower()

        print(f"[{i}/{len(test_cases)}] {case_id} | role: {role} | truth: {truth.upper()}", end=" ... ")

        t0 = time.time()
        try:
            output = decide(
                role, cv_text, jd_text, skills_cv, skills_jd,
                exclude_case_id=case_id,
            )
            pred       = output.get("decision", "unknown").lower()
            confidence = output.get("confidence", "unknown")
            elapsed    = round(time.time() - t0, 1)
            correct    = pred == truth

            print(f"pred: {pred.upper()} | {'✓' if correct else '✗'} | {elapsed}s")

            results.append({
                "case_id":      case_id,
                "role":         role,
                "ground_truth": truth,
                "prediction":   pred,
                "confidence":   confidence,
                "correct":      correct,
                "elapsed_s":    elapsed,
                "key_strengths": output.get("key_strengths", []),
                "key_gaps":      output.get("key_gaps", []),
                "reasoning":    output.get("reasoning", ""),
                "parse_error":  output.get("parse_error", False),
            })

        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            print(f"ERROR: {e}")
            results.append({
                "case_id":      case_id,
                "role":         role,
                "ground_truth": truth,
                "prediction":   "error",
                "correct":      False,
                "elapsed_s":    elapsed,
                "error":        str(e),
            })

    # Compute metrics (exclude error rows)
    valid = [r for r in results if r["prediction"] not in ("error", "unknown")]
    metrics = compute_metrics(valid)
    avg_time = round(sum(r["elapsed_s"] for r in results) / len(results), 1)

    # Cases dropped before scoring, and cases where the LLM's JSON had to be
    # recovered by the text fallback. Both silently shrink the sample, so they
    # belong in the summary rather than only in the per-case rows.
    n_dropped      = len(results) - len(valid)
    n_parse_errors = sum(1 for r in results if r.get("parse_error"))

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Cases evaluated : {metrics['total_cases']}"
          f"{f' ({n_dropped} dropped)' if n_dropped else ''}")
    print(f"  Accuracy        : {metrics['accuracy']:.1%}")
    print(f"  Precision       : {metrics['precision']:.1%}")
    print(f"  Recall          : {metrics['recall']:.1%}")
    print(f"  F1 Score        : {metrics['f1']:.1%}")
    print(f"  Predicted SELECT: {metrics['pred_select_rate']:.1%} of cases")
    print(f"  JSON parse fails: {n_parse_errors}")
    print(f"  Avg. time/case  : {avg_time}s")
    print()
    cm = metrics["confusion_matrix"]
    print(f"  Confusion matrix:")
    for label, count in cm.items():
        print(f"    {label}: {count}")
    print()
    print("  Reference lines (see EVALUATION_AND_APPROACH_PLAN.md §0.1):")
    print("    Random / majority class      : 50.0%")
    print("    Supervised ceiling (TF-IDF+LR): 58.2%")
    print("=" * 50)

    # Save results
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "n_cases": len(test_cases),
            "seed":    seed,
            "model":   "llama3.1:8b",
            "retrieval": "faiss_two_view_mpnet",
            "max_per_class": 5,
            "index_leak_fixed": True,   # decision_reason removed from Step 3 case view
        },
        "metrics":  metrics,
        "n_dropped": n_dropped,
        "n_parse_errors": n_parse_errors,
        "avg_time_s": avg_time,
        "results":  results,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=20, help="Number of test cases (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out",  type=str, default=str(RESULTS_PATH),
                        help="Where to write results (default: data/processed/eval_results.json). "
                             "Use a distinct path per run so earlier results are not overwritten.")
    args = parser.parse_args()

    main(n=args.n, seed=args.seed, out=Path(args.out))
