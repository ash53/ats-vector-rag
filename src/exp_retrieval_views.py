"""
EXPERIMENT — is the two-view embedding worth it? (Layer 1, no LLM)

Step 3 embeds each case twice and concatenates:

  case view   role + resume + job description   "have we seen this case before?"
  skill view  candidate skills, required skills, explicit gaps

The concatenation is the distinctive design choice of this pipeline and it has
never been validated. This measures it at the retrieval layer only — no LLM, so
nothing is obscured by the decision stage and the noisy label cannot interfere.

No re-encoding is needed. The stored vectors are [case ‖ skill], each half
L2-normalised before the hstack, so the two single-view indices are recovered by
slicing what is already on disk.

WHAT RETRIEVAL WOULD HAVE TO ACHIEVE TO BE USEFUL

The pipeline shows the LLM ten historical cases and their outcomes, as evidence
for a new decision. That only helps if retrieved cases tend to share the query's
outcome. If the same-decision rate is ~50% on a balanced corpus, the exemplars
carry no information about the query no matter how the embedding is built, and
no amount of retrieval engineering can fix it. That number is the point of this
script; it explains the RAG-vs-zero-shot null (eval_controls_n300.json)
mechanically rather than just observing it.

Metrics per view, leave-one-out over the standard seed-42 sample:

  same-decision@10   share of retrieved cases whose outcome matches the query.
                     50% = the exemplars say nothing. THE headline number.
  same-role@10       share from the query's own role.
  skill Jaccard      mean skill overlap between query and retrieved CVs.
  spread             max-min similarity across the top 10 — how sharply the
                     index separates the neighbours it returns.
  diversity          mean pairwise cosine among retrieved (lower = more varied).

Usage:
    python src/exp_retrieval_views.py            # n=300, seed 42
    python src/exp_retrieval_views.py --n 1000
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import argparse
from pathlib import Path

import numpy as np
import faiss

from step6_evaluate import load_cases, stratified_sample

INDEX_PATH = Path("data/processed/faiss_index.bin")
META_PATH = Path("data/processed/faiss_metadata.json")
CASES_PATH = Path("data/processed/cases_stage2.jsonl")
RESULTS_PATH = Path("data/processed/eval_retrieval_views.json")


def load_views():
    """Recover the case-only and skill-only matrices from the stored index."""
    index = faiss.read_index(str(INDEX_PATH))
    n, d = index.ntotal, index.d
    full = index.reconstruct_n(0, n).astype("float32")
    half = d // 2
    case_v, skill_v = full[:, :half].copy(), full[:, half:].copy()

    # Each half was normalised before hstack; confirm rather than assume.
    for name, m in [("case", case_v), ("skill", skill_v)]:
        norms = np.linalg.norm(m, axis=1)
        print(f"  {name} view: dim {m.shape[1]}, "
              f"norm mean {norms.mean():.4f} (expected 1.0)")

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return {"case_only": case_v, "skill_only": skill_v, "two_view": full}, metadata


def evaluate_view(vectors, metadata, cases_by_id, test_cases, k=10):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    pos_by_id = {m["case_id"]: i for i, m in enumerate(metadata)}

    same_dec, same_role, jaccard, spreads, diversity = [], [], [], [], []

    for c in test_cases:
        qi = pos_by_id[c["case_id"]]
        q = vectors[qi:qi + 1]
        scores, idx = index.search(q, k + 1)

        # leave-one-out: drop the query itself
        keep = [(i, s) for i, s in zip(idx[0], scores[0]) if i != qi][:k]
        if not keep:
            continue
        ids = [metadata[i]["case_id"] for i, _ in keep]
        sims = [s for _, s in keep]

        truth = c["decision"].lower()
        same_dec.append(np.mean([metadata[i]["decision"].lower() == truth
                                 for i, _ in keep]))
        same_role.append(np.mean([metadata[i]["role"] == c["role"]
                                  for i, _ in keep]))

        qs = set(c["entities"]["skills_cv"])
        js = []
        for cid in ids:
            rs = set(cases_by_id[cid]["entities"]["skills_cv"])
            js.append(len(qs & rs) / len(qs | rs) if (qs | rs) else 0.0)
        jaccard.append(np.mean(js))

        spreads.append(max(sims) - min(sims))

        sub = vectors[[i for i, _ in keep]]
        sub = sub / np.linalg.norm(sub, axis=1, keepdims=True)
        pair = sub @ sub.T
        m = ~np.eye(len(sub), dtype=bool)
        diversity.append(float(pair[m].mean()))

    return {
        "same_decision_at_k": round(float(np.mean(same_dec)), 4),
        "same_role_at_k": round(float(np.mean(same_role)), 4),
        "skill_jaccard": round(float(np.mean(jaccard)), 4),
        "similarity_spread": round(float(np.mean(spreads)), 4),
        "exemplar_diversity": round(float(np.mean(diversity)), 4),
        "n_queries": len(same_dec),
    }


def main(n=300, seed=42, k=10, out=RESULTS_PATH):
    print("Loading and splitting the index...")
    views, metadata = load_views()

    cases = load_cases(CASES_PATH)
    cases_by_id = {c["case_id"]: c for c in cases}
    test_cases = stratified_sample(cases, n, seed)
    print(f"\n{len(test_cases)} queries (seed={seed}), k={k}\n")

    results = {}
    for name, vecs in views.items():
        results[name] = evaluate_view(vecs, metadata, cases_by_id, test_cases, k)
        r = results[name]
        print(f"{name:12s} same-decision {r['same_decision_at_k']:.1%} | "
              f"same-role {r['same_role_at_k']:.1%} | "
              f"skill-Jaccard {r['skill_jaccard']:.3f} | "
              f"spread {r['similarity_spread']:.4f} | "
              f"diversity {r['exemplar_diversity']:.4f}")

    print("\n" + "=" * 72)
    best = max(results, key=lambda v: results[v]["same_decision_at_k"])
    span = (max(r["same_decision_at_k"] for r in results.values()) -
            min(r["same_decision_at_k"] for r in results.values()))
    print(f"Best same-decision rate: {best} "
          f"({results[best]['same_decision_at_k']:.1%}); "
          f"spread across views {span:.1%}")
    print("50% means the retrieved outcomes carry no information about the query.")
    print("=" * 72)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"config": {"n": n, "seed": seed, "k": k}, "views": results},
                  f, indent=2)
    tmp.replace(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=str, default=str(RESULTS_PATH))
    a = p.parse_args()
    main(n=a.n, seed=a.seed, k=a.k, out=Path(a.out))
