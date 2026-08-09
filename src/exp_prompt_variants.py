"""
EXPERIMENT — Prompt variants for the decision layer

The n=40 baseline run (eval_results_n40.json) predicted SELECT on 85% of cases
and scored 45.0% accuracy / 59.3% F1 — worse than a classifier that answers
"select" unconditionally (50.0% / 66.7%) on the same balanced sample. Amol's
GraphRAG pipeline shows the same failure with an identical 80.0% recall, so the
problem is in the decision layer, not in either retrieval implementation.

This script isolates the prompt. Retrieval runs ONCE per case and every variant
sees exactly the same evidence, so any difference in the numbers is caused by
the prompt alone. All variants run on the same cases as the baseline
(stratified_sample(n=40, seed=42)), which makes the comparison paired.

Usage:
    python src/exp_prompt_variants.py                     # all variants, 40 cases
    python src/exp_prompt_variants.py --n 20 --variants v2,v3
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import time
import argparse
from pathlib import Path

import step5_llm_decision as s5
from step6_evaluate import load_cases, stratified_sample, compute_metrics

CASES_PATH = Path("data/processed/cases_stage2.jsonl")
RESULTS_PATH = Path("data/processed/eval_prompt_variants.json")

JSON_CONTRACT = """{
    "decision": "select" or "reject",
    "confidence": "high" or "medium" or "low",
    "key_strengths": ["strength1", "strength2"],
    "key_gaps": ["gap1", "gap2"],
    "reasoning": "Your explanation here"
}"""


def _profile(role, cv_text, jd_text, skills_cv, skills_jd):
    """Shared candidate/JD block — identical across variants."""
    missing = sorted(set(skills_jd) - set(skills_cv))
    return f"""## JOB DESCRIPTION
{jd_text}

## CANDIDATE PROFILE
- Role applied for: {role}
- Resume: {cv_text[:1500]}{"..." if len(cv_text) > 1500 else ""}
- Extracted skills: {", ".join(skills_cv) or "none detected"}
- Job-required skills: {", ".join(skills_jd) or "none detected"}
- Skill gaps: {", ".join(missing) or "none"}"""


def _exemplars(evidence):
    return "".join(s5._format_exemplar(c, i) for i, c in enumerate(evidence, 1))


# ---------------------------------------------------------------------------
# v0 — baseline (the current Step 5 prompt, unchanged)
# ---------------------------------------------------------------------------
def v0_baseline(role, cv_text, jd_text, skills_cv, skills_jd, evidence):
    return s5.build_prompt(role, cv_text, jd_text, skills_cv, skills_jd, evidence)


# ---------------------------------------------------------------------------
# v1 — reason before deciding
# ---------------------------------------------------------------------------
# The baseline asks for the DECISION as task 1 and the REASONING as task 5, and
# puts "decision" first in the JSON. A non-thinking model therefore emits its
# verdict before generating a single token of analysis, and the rest of the
# output is written to justify a commitment already made. This variant changes
# only the order — analysis first, verdict last.
def v1_reason_first(role, cv_text, jd_text, skills_cv, skills_jd, evidence):
    return f"""You are an expert HR analyst making data-driven hiring decisions.
Evaluate the candidate's CV against the job description using the historical reference cases below.

{_profile(role, cv_text, jd_text, skills_cv, skills_jd)}

## HISTORICAL REFERENCE CASES (vector-retrieved, stratified by outcome)
{_exemplars(evidence)}

## YOUR TASK
Work through these in order. Do not decide until step 4.

1. KEY STRENGTHS: 2-3 strengths relative to the job requirements
2. KEY GAPS: 2-3 gaps or concerns relative to the job requirements
3. REASONING: 2-3 sentences weighing the strengths against the gaps
4. DECISION: only now, SELECT or REJECT
5. CONFIDENCE: HIGH / MEDIUM / LOW

Respond in this exact JSON format only - no extra text. Note the key order:
{{
    "key_strengths": ["strength1", "strength2"],
    "key_gaps": ["gap1", "gap2"],
    "reasoning": "Your explanation here",
    "decision": "select" or "reject",
    "confidence": "high" or "medium" or "low"
}}

JSON Response:"""


# ---------------------------------------------------------------------------
# v2 — base rate stated, exemplar balance explained, no default
# ---------------------------------------------------------------------------
# Two fixes. (a) The model is never told the outcome base rate, and defaults to
# approving. (b) The 5-select/5-reject exemplar split is imposed by the
# retriever, not observed in the data — a model reading it as evidence sees a
# balanced-to-positive signal on every single case. Saying so explicitly stops
# the exemplar mix being read as support for approval.
def v2_base_rate(role, cv_text, jd_text, skills_cv, skills_jd, evidence):
    return f"""You are an expert HR analyst making data-driven hiring decisions.
Evaluate the candidate's CV against the job description using the historical reference cases below.

{_profile(role, cv_text, jd_text, skills_cv, skills_jd)}

## HISTORICAL REFERENCE CASES
{_exemplars(evidence)}

## IMPORTANT CONTEXT BEFORE YOU DECIDE
- In this hiring pipeline roughly 50% of candidates are REJECTED. Rejection is
  the normal outcome, not the exception.
- The reference cases above were deliberately balanced by the retrieval system:
  it always returns 5 selected and 5 rejected cases. That balance is an artifact
  of how they were fetched and tells you NOTHING about this candidate. Do not
  read it as evidence either way.
- A candidate who merely looks plausible for the role is not a SELECT. Select
  only if the evidence positively supports it.
- Before answering SELECT, check the gaps you identified. If any of them would
  materially impair performance in this role, answer REJECT.

## YOUR TASK
1. KEY STRENGTHS: 2-3 strengths supported by the resume
2. KEY GAPS: 2-3 gaps or concerns
3. REASONING: 2-3 sentences, explicitly addressing whether the gaps are disqualifying
4. DECISION: SELECT or REJECT
5. CONFIDENCE: HIGH / MEDIUM / LOW

Respond in this exact JSON format only - no extra text:
{JSON_CONTRACT}

JSON Response:"""


# ---------------------------------------------------------------------------
# v3 — requirement checklist (miniature of the decomposition approach)
# ---------------------------------------------------------------------------
# Instead of one holistic judgement, the model checks each job requirement
# against the resume and applies an explicit aggregation rule. The per
# requirement verdicts are individually checkable against the CV, which is what
# makes faithfulness measurable later. Combines the framing fixes from v1+v2.
def v3_checklist(role, cv_text, jd_text, skills_cv, skills_jd, evidence):
    return f"""You are an expert HR analyst making data-driven hiring decisions.

{_profile(role, cv_text, jd_text, skills_cv, skills_jd)}

## HISTORICAL REFERENCE CASES (for calibration only)
{_exemplars(evidence)}

## IMPORTANT CONTEXT
- Roughly 50% of candidates in this pipeline are REJECTED. Rejection is normal.
- The reference cases are always returned 5 selected / 5 rejected by the
  retrieval system. That balance is an artifact and carries no signal.

## YOUR TASK — follow the procedure, do not skip ahead
1. List each requirement of the job (from the job description and the
   job-required skills). Aim for 4-6 requirements.
2. For each requirement, mark it MET or MISSING, and quote the evidence from
   the resume that supports MET. If you cannot quote evidence, it is MISSING.
3. Count how many are MET.
4. Apply this rule: if fewer than half the requirements are MET, decision is
   REJECT. Otherwise decision is SELECT.
5. State your confidence.

Respond in this exact JSON format only - no extra text:
{{
    "requirements": [
        {{"requirement": "...", "status": "met" or "missing", "evidence": "quote from resume, or empty"}}
    ],
    "n_met": 0,
    "n_total": 0,
    "key_strengths": ["strength1", "strength2"],
    "key_gaps": ["gap1", "gap2"],
    "reasoning": "Your explanation here",
    "decision": "select" or "reject",
    "confidence": "high" or "medium" or "low"
}}

JSON Response:"""


VARIANTS = {
    "v0": ("baseline (current prompt)", v0_baseline),
    "v1": ("reason before deciding", v1_reason_first),
    "v2": ("base rate + no default", v2_base_rate),
    "v3": ("requirement checklist", v3_checklist),
}


def main(n=40, seed=42, variant_keys=None):
    variant_keys = variant_keys or list(VARIANTS)

    index, metadata, cases_by_id = s5._load_artifacts()
    model = s5._load_model()

    all_cases = load_cases(CASES_PATH)
    test_cases = stratified_sample(all_cases, n, seed)
    print(f"{len(test_cases)} cases (seed={seed}), variants: {', '.join(variant_keys)}\n")

    # Retrieve once per case; every variant reuses the identical evidence.
    print("Retrieving evidence...", end=" ", flush=True)
    t0 = time.time()
    evidence_by_case = {}
    for c in test_cases:
        qv = s5.embed_query(model, c["role"], c["cv_text"], c["jd_text"],
                            c["entities"]["skills_cv"], c["entities"]["skills_jd"])
        retrieved = s5.retrieve_similar_cases(index, metadata, cases_by_id, qv,
                                              k=20, exclude_case_id=c["case_id"])
        evidence_by_case[c["case_id"]] = s5.build_evidence_chunks(
            s5.stratify_results(retrieved, max_per_class=5))
    print(f"{time.time()-t0:.0f}s\n")

    all_results = {}

    for key in variant_keys:
        label, builder = VARIANTS[key]
        print(f"=== {key}: {label} ===")
        rows = []

        for i, c in enumerate(test_cases, 1):
            truth = c["decision"].lower()
            prompt = builder(c["role"], c["cv_text"], c["jd_text"],
                             c["entities"]["skills_cv"], c["entities"]["skills_jd"],
                             evidence_by_case[c["case_id"]])
            t0 = time.time()
            out = s5.parse_decision(s5.call_llm(prompt))
            elapsed = round(time.time() - t0, 1)
            pred = str(out.get("decision", "unknown")).lower()

            rows.append({
                "case_id": c["case_id"],
                "role": c["role"],
                "ground_truth": truth,
                "prediction": pred,
                "confidence": out.get("confidence", "unknown"),
                "correct": pred == truth,
                "elapsed_s": elapsed,
                "key_strengths": out.get("key_strengths", []),
                "key_gaps": out.get("key_gaps", []),
                "reasoning": out.get("reasoning", ""),
                "requirements": out.get("requirements", []),
                "n_met": out.get("n_met"),
                "n_total": out.get("n_total"),
                "parse_error": out.get("parse_error", False),
            })
            print(f"  [{i}/{len(test_cases)}] {truth[:3]}->{pred[:3]} "
                  f"{'ok' if pred == truth else 'X'} {elapsed}s", flush=True)

        valid = [r for r in rows if r["prediction"] not in ("error", "unknown")]
        m = compute_metrics(valid)
        m["n_parse_errors"] = sum(1 for r in rows if r["parse_error"])
        m["avg_time_s"] = round(sum(r["elapsed_s"] for r in rows) / len(rows), 1)
        all_results[key] = {"label": label, "metrics": m, "results": rows}

        print(f"  -> acc {m['accuracy']:.1%} | P {m['precision']:.1%} | "
              f"R {m['recall']:.1%} | F1 {m['f1']:.1%} | "
              f"select-rate {m['pred_select_rate']:.1%} | "
              f"parse-fails {m['n_parse_errors']}\n", flush=True)

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"config": {"n": n, "seed": seed, "model": s5.MODEL_NAME},
                       "variants": all_results}, f, indent=2)

    # v3-rule — derived, no extra LLM calls.
    # In the v3 smoke test the model returned n_met 1 of 2 and still answered
    # REJECT, contradicting the aggregation rule it was given. The fix is to
    # stop asking the LLM to aggregate: keep its per-requirement judgements and
    # apply the threshold in code. Recomputed here from the saved counts.
    if "v3" in all_results:
        rows = []
        for r in all_results["v3"]["results"]:
            n_total = r.get("n_total") or len(r.get("requirements") or [])
            n_met = r.get("n_met")
            if n_met is None:
                n_met = sum(1 for q in (r.get("requirements") or [])
                            if str(q.get("status", "")).lower() == "met")
            if not n_total:
                pred = r["prediction"]          # no checklist to score; keep as-is
            else:
                pred = "select" if n_met >= n_total / 2 else "reject"
            rows.append({**r, "prediction": pred,
                         "correct": pred == r["ground_truth"]})

        valid = [r for r in rows if r["prediction"] not in ("error", "unknown")]
        m = compute_metrics(valid)
        m["n_parse_errors"] = all_results["v3"]["metrics"]["n_parse_errors"]
        m["avg_time_s"] = 0.0
        n_overridden = sum(1 for a, b in zip(rows, all_results["v3"]["results"])
                           if a["prediction"] != b["prediction"])
        m["n_llm_rule_violations"] = n_overridden
        all_results["v3-rule"] = {
            "label": "checklist + rule in code (derived)",
            "metrics": m, "results": rows,
        }
        variant_keys = list(variant_keys) + ["v3-rule"]
        print(f"=== v3-rule (derived): aggregation applied in code ===")
        print(f"  LLM decisions overridden by its own counts: {n_overridden}/{len(rows)}")
        print(f"  -> acc {m['accuracy']:.1%} | P {m['precision']:.1%} | "
              f"R {m['recall']:.1%} | F1 {m['f1']:.1%} | "
              f"select-rate {m['pred_select_rate']:.1%}\n")

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"config": {"n": n, "seed": seed, "model": s5.MODEL_NAME},
                       "variants": all_results}, f, indent=2)

    # Degenerate reference lines on the same sample
    n_sel = sum(1 for c in test_cases if c["decision"].lower() == "select")
    always_p = n_sel / len(test_cases)
    always_f1 = 2 * always_p / (always_p + 1)

    print("=" * 78)
    print(f"{'variant':<28} {'acc':>7} {'prec':>7} {'rec':>7} {'F1':>7} {'sel-rate':>9}")
    print("-" * 78)
    print(f"{'always-select (degenerate)':<28} {always_p:>7.1%} {always_p:>7.1%} "
          f"{1.0:>7.1%} {always_f1:>7.1%} {1.0:>9.1%}")
    for key in variant_keys:
        m = all_results[key]["metrics"]
        print(f"{key + ' ' + all_results[key]['label']:<28} {m['accuracy']:>7.1%} "
              f"{m['precision']:>7.1%} {m['recall']:>7.1%} {m['f1']:>7.1%} "
              f"{m['pred_select_rate']:>9.1%}")
    print("=" * 78)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", type=str, default=None,
                   help="comma-separated subset, e.g. v1,v2")
    a = p.parse_args()
    keys = a.variants.split(",") if a.variants else None
    main(n=a.n, seed=a.seed, variant_keys=keys)
