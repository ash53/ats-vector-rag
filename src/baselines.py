"""
BASELINES — the reference lines every RAG result must be read against

Three non-LLM systems, all scored on exactly the same test sample as
step6_evaluate.py (stratified_sample(n, seed)) so the numbers are directly
comparable to the RAG pipeline.

1. always-select   Degenerate. Answers "select" every time. On a balanced test
                   set this scores 50% accuracy and 66.7% F1 — which the n=40
                   RAG run (45.0% / 59.3%) did not beat. Any system that fails
                   to clear this line is not making a decision.

2. keyword         The traditional ATS the proposal requires us to benchmark
                   against: TF-IDF cosine similarity between CV and JD, with a
                   threshold. No embeddings, no LLM. The threshold is tuned on
                   the training split only — never on the test sample.

3. supervised      TF-IDF + logistic regression trained on every case outside
                   the test sample. Not a system we would ship: it is the
                   practical ceiling of what these features contain about the
                   label. Measured at 58.2% accuracy / 0.639 AUC in 5-fold CV
                   over the full dataset (see EVALUATION_AND_APPROACH_PLAN.md).

Usage:
    python src/baselines.py                  # n=300, seed=42
    python src/baselines.py --n 40 --seed 42 # match the n=40 RAG run
"""

import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from step6_evaluate import load_cases, stratified_sample, compute_metrics

CASES_PATH = Path("data/processed/cases_stage2.jsonl")
RESULTS_PATH = Path("data/processed/eval_baselines.json")


def _rows(cases, preds):
    return [{"case_id": c["case_id"], "role": c["role"],
             "ground_truth": c["decision"].lower(), "prediction": p,
             "correct": p == c["decision"].lower()}
            for c, p in zip(cases, preds)]


def always_select(test_cases):
    return _rows(test_cases, ["select"] * len(test_cases))


def keyword_baseline(train_cases, test_cases):
    """TF-IDF cosine similarity between CV and JD, thresholded.

    The threshold is chosen on the training split by sweeping candidate cut-offs
    and keeping the one with the best training accuracy. Tuning it on the test
    sample would inflate the baseline we are trying to beat.
    """
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True)
    vec.fit([c["cv_text"] + " " + c["jd_text"] for c in train_cases])

    def sims(cases):
        cv = vec.transform([c["cv_text"] for c in cases])
        jd = vec.transform([c["jd_text"] for c in cases])
        # rows are L2-normalized by TfidfVectorizer, so the dot product is cosine
        return np.asarray(cv.multiply(jd).sum(axis=1)).ravel()

    train_sim = sims(train_cases)
    train_y = np.array([1 if c["decision"].lower() == "select" else 0
                        for c in train_cases])

    best_t, best_acc = 0.0, -1.0
    for t in np.quantile(train_sim, np.linspace(0.05, 0.95, 91)):
        acc = ((train_sim >= t).astype(int) == train_y).mean()
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)

    test_sim = sims(test_cases)
    preds = ["select" if s >= best_t else "reject" for s in test_sim]

    test_y = [1 if c["decision"].lower() == "select" else 0 for c in test_cases]
    auc = roc_auc_score(test_y, test_sim) if len(set(test_y)) > 1 else float("nan")
    return _rows(test_cases, preds), {"threshold": best_t,
                                      "train_accuracy": round(best_acc, 4),
                                      "test_auc": round(float(auc), 4)}


def supervised_baseline(train_cases, test_cases):
    """TF-IDF + logistic regression on CV + JD text. The learnable ceiling."""
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform([c["cv_text"] + " " + c["jd_text"] for c in train_cases])
    ytr = [1 if c["decision"].lower() == "select" else 0 for c in train_cases]

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)

    Xte = vec.transform([c["cv_text"] + " " + c["jd_text"] for c in test_cases])
    proba = clf.predict_proba(Xte)[:, 1]
    preds = ["select" if p >= 0.5 else "reject" for p in proba]

    yte = [1 if c["decision"].lower() == "select" else 0 for c in test_cases]
    auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan")
    return _rows(test_cases, preds), {"test_auc": round(float(auc), 4),
                                      "n_train": len(train_cases)}


def main(n=300, seed=42, out=RESULTS_PATH):
    all_cases = load_cases(CASES_PATH)
    test_cases = stratified_sample(all_cases, n, seed)
    test_ids = {c["case_id"] for c in test_cases}
    train_cases = [c for c in all_cases if c["case_id"] not in test_ids]

    print(f"test: {len(test_cases)} cases (n={n}, seed={seed}) | "
          f"train: {len(train_cases)}\n")

    out_data = {"config": {"n": n, "seed": seed}, "baselines": {}}

    for name, fn in [("always-select", lambda: (always_select(test_cases), {})),
                     ("keyword-tfidf", lambda: keyword_baseline(train_cases, test_cases)),
                     ("supervised-logreg", lambda: supervised_baseline(train_cases, test_cases))]:
        rows, extra = fn()
        m = compute_metrics(rows)
        m.update(extra)
        out_data["baselines"][name] = {"metrics": m, "results": rows}
        print(f"{name:<20} acc {m['accuracy']:.1%} | P {m['precision']:.1%} | "
              f"R {m['recall']:.1%} | F1 {m['f1']:.1%} | "
              f"select-rate {m['pred_select_rate']:.1%}"
              + (f" | AUC {m['test_auc']:.3f}" if "test_auc" in m else ""))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(RESULTS_PATH))
    a = p.parse_args()
    main(n=a.n, seed=a.seed, out=Path(a.out))
