"""
EXPERIMENT — what actually drives the decision: the prompt, or the evidence?

Two families of experiment share this harness. Both run on the same cases as
Step 6 (stratified_sample(n, seed)), so every comparison is paired, and all
conditions reuse identical evidence built once per case — only the thing under
test changes.

  v0-v3      hold the EVIDENCE fixed, vary the PROMPT.
  c_*        hold the PROMPT fixed, vary the EVIDENCE (retrieved / random
             same-role / random any-role / none). These are the controls that
             isolate whether retrieval contributes anything at all — see the
             CONTROLS section below.

What is known so far (all seed 42, leak-free index, llama3.1:8b):

  RAG at n=300      55.3% accuracy, 82.7% select rate. Beats always-select
                    (50.0%) on a paired McNemar test, p=0.037, and is
                    indistinguishable from TF-IDF + logistic regression
                    (53.7%, p=0.75).
  Prompt variants   ran at n=40 BEFORE temperature was pinned, so their
                    ranking is not trustworthy: the identical prompt on the
                    identical cases scored 45.0% and then 50.0%. What survived
                    replication is that the select rate is movable (90% -> 47%)
                    without accuracy improving, that the model contradicted its
                    own stated aggregation rule on 30% of cases, and that its
                    confidence is uncorrelated with being right.

Runs are greedy and seeded (step5.TEMPERATURE = 0), so results reproduce.

Usage:
    python src/exp_prompt_variants.py --n 300 --variants v0,v3 --out data/processed/eval_prompts_n300.json
    python src/exp_prompt_variants.py --n 300 \\
        --variants c_retrieved,c_random_role,c_random_corpus,c_zeroshot \\
        --out data/processed/eval_controls_n300.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import time
import random
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


# ---------------------------------------------------------------------------
# CONTROLS — does the retrieval contribute anything?
# ---------------------------------------------------------------------------
# Every number measured so far compares RAG variants against non-RAG baselines.
# Nothing has isolated the R in RAG. These controls hold the prompt fixed (the
# v0 wording) and change only what evidence goes into it:
#
#   c_retrieved     top-20 by similarity, stratified 5/5    (the real system)
#   c_random_role   10 random cases from the same role      (is the RANKING
#                                                            worth anything, or
#                                                            is role-matching
#                                                            the whole effect?)
#   c_random_corpus 10 random cases from anywhere           (does retrieval
#                                                            matter at all?)
#   c_zeroshot      no exemplars at all                     (does RAG beat just
#                                                            asking the model?)
#
# Retrieval on this index returns 20/20 same-role exemplars spanning 0.064 of
# similarity, so the ranking may be carrying no information. These four numbers
# settle it.
def c_zeroshot(role, cv_text, jd_text, skills_cv, skills_jd, evidence):
    """Identical to v0 minus the historical cases section."""
    return f"""You are an expert HR analyst making data-driven hiring decisions.
Evaluate the candidate's CV against the job description.

{_profile(role, cv_text, jd_text, skills_cv, skills_jd)}

## YOUR TASK
Using the job description and candidate profile:

1. DECISION: Should this candidate be SELECTED or REJECTED?
2. CONFIDENCE: Rate your confidence (HIGH / MEDIUM / LOW)
3. KEY STRENGTHS: 2-3 strengths supporting selection
4. KEY GAPS: 2-3 gaps or concerns
5. REASONING: 2-3 sentences on how the candidate fits the role

Respond in this exact JSON format only - no extra text:
{JSON_CONTRACT}

JSON Response:"""


# key -> (label, prompt builder, evidence mode)
VARIANTS = {
    "v0": ("baseline (current prompt)", v0_baseline, "retrieved"),
    "v1": ("reason before deciding", v1_reason_first, "retrieved"),
    "v2": ("base rate + no default", v2_base_rate, "retrieved"),
    "v3": ("requirement checklist", v3_checklist, "retrieved"),
    "c_retrieved":     ("CONTROL retrieved exemplars", v0_baseline, "retrieved"),
    "c_random_role":   ("CONTROL random, same role", v0_baseline, "random_role"),
    "c_random_corpus": ("CONTROL random, any role", v0_baseline, "random_corpus"),
    "c_zeroshot":      ("CONTROL no exemplars", c_zeroshot, "none"),
}


def _random_evidence(index, metadata, cases_by_id, pos_by_id, query_vec,
                     query_case, rng, same_role, max_per_class=5):
    """10 random cases (5 select / 5 reject) instead of the retrieved ones.

    The similarity scores printed in the prompt are the REAL cosine similarities
    of the sampled cases to the query, reconstructed from the index — so the
    only thing that differs from the retrieved condition is which cases were
    chosen, not how they are described.
    """
    pool = [m for m in metadata if m["case_id"] != query_case["case_id"]]
    if same_role:
        pool = [m for m in pool if m["role"] == query_case["role"]] or pool

    picked = []
    for decision in ["select", "reject"]:
        bucket = [m for m in pool if m["decision"] == decision]
        picked += rng.sample(bucket, min(max_per_class, len(bucket)))

    q = query_vec[0]
    out = []
    for m in picked:
        vec = index.reconstruct(pos_by_id[m["case_id"]])
        out.append({"case_id": m["case_id"], "decision": m["decision"],
                    "similarity_score": float(vec @ q), "role": m["role"],
                    **{k: cases_by_id[m["case_id"]]["entities"][k]
                       for k in ["skills_cv", "skills_jd", "degrees", "certifications"]},
                    "decision_reason": cases_by_id[m["case_id"]]["decision_reason"]})
    return out


def _save(path, n, seed, all_results):
    """Checkpoint after every condition, atomically, and never let a write
    failure kill the run — each condition costs an hour of LLM time."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"config": {"n": n, "seed": seed, "model": s5.MODEL_NAME,
                                  "temperature": s5.TEMPERATURE},
                       "variants": all_results}, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        print(f"  [WARNING] could not save results to {path}: {e!r}", flush=True)


def main(n=40, seed=42, variant_keys=None, out=RESULTS_PATH):
    out = Path(out)
    variant_keys = variant_keys or list(VARIANTS)
    modes = {VARIANTS[k][2] for k in variant_keys}

    index, metadata, cases_by_id = s5._load_artifacts()
    model = s5._load_model()
    pos_by_id = {m["case_id"]: i for i, m in enumerate(metadata)}

    all_cases = load_cases(CASES_PATH)
    test_cases = stratified_sample(all_cases, n, seed)
    print(f"{len(test_cases)} cases (seed={seed}) | temperature={s5.TEMPERATURE} "
          f"| variants: {', '.join(variant_keys)}\n")

    # Built once per case; every variant reuses the identical evidence, so the
    # only thing that differs between two runs is the thing being tested.
    print("Building evidence...", end=" ", flush=True)
    t0 = time.time()
    rng = random.Random(seed)
    evidence = {m: {} for m in modes}
    for c in test_cases:
        qv = s5.embed_query(model, c["role"], c["cv_text"], c["jd_text"],
                            c["entities"]["skills_cv"], c["entities"]["skills_jd"])
        if "retrieved" in modes:
            retrieved = s5.retrieve_similar_cases(index, metadata, cases_by_id, qv,
                                                  k=20, exclude_case_id=c["case_id"])
            evidence["retrieved"][c["case_id"]] = s5.build_evidence_chunks(
                s5.stratify_results(retrieved, max_per_class=5))
        for mode, same_role in [("random_role", True), ("random_corpus", False)]:
            if mode in modes:
                evidence[mode][c["case_id"]] = _random_evidence(
                    index, metadata, cases_by_id, pos_by_id, qv, c, rng, same_role)
        if "none" in modes:
            evidence["none"][c["case_id"]] = []
    print(f"{time.time()-t0:.0f}s\n")

    evidence_by_case = evidence.get("retrieved", {})

    all_results = {}

    for key in variant_keys:
        label, builder, mode = VARIANTS[key]
        print(f"=== {key}: {label} [{mode}] ===")
        rows = []

        for i, c in enumerate(test_cases, 1):
            truth = c["decision"].lower()
            prompt = builder(c["role"], c["cv_text"], c["jd_text"],
                             c["entities"]["skills_cv"], c["entities"]["skills_jd"],
                             evidence[mode][c["case_id"]])
            t0 = time.time()
            # NOT `out` — that is the results path, and shadowing it here cost a
            # completed 300-case condition: the run reached the save at the end
            # of the first variant, called open() on a decision dict and died.
            decision = s5.parse_decision(s5.call_llm(prompt))
            elapsed = round(time.time() - t0, 1)
            pred = str(decision.get("decision", "unknown")).lower()

            rows.append({
                "case_id": c["case_id"],
                "role": c["role"],
                "ground_truth": truth,
                "prediction": pred,
                "confidence": decision.get("confidence", "unknown"),
                "correct": pred == truth,
                "elapsed_s": elapsed,
                "key_strengths": decision.get("key_strengths", []),
                "key_gaps": decision.get("key_gaps", []),
                "reasoning": decision.get("reasoning", ""),
                "requirements": decision.get("requirements", []),
                "n_met": decision.get("n_met"),
                "n_total": decision.get("n_total"),
                "parse_error": decision.get("parse_error", False),
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

        _save(out, n, seed, all_results)

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

        _save(out, n, seed, all_results)

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
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", type=str, default=None,
                   help="comma-separated subset, e.g. v1,v2")
    p.add_argument("--out", type=str, default=str(RESULTS_PATH),
                   help="results path — use a distinct one per run so earlier "
                        "results are not overwritten")
    a = p.parse_args()
    keys = a.variants.split(",") if a.variants else None
    main(n=a.n, seed=a.seed, variant_keys=keys, out=Path(a.out))
