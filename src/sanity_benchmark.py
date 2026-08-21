"""
CONTROLLED SANITY BENCHMARK — a task where the right answer is knowable

Every result in this project is bounded by a dataset whose labels are largely
uncorrelated with the candidate data: the reason for each decision is
unpredictable from the CV (10.5% against a 10.4% majority baseline), the job
descriptions average 53 words and only 21.7% state a requirement, and a fully
supervised classifier trained on the labels reaches 58.2%. Everything we build
scores in the mid-50s, and we cannot tell from those numbers alone whether the
pipeline is broken or the data is unlearnable.

This benchmark separates the two. It generates cases where the label IS a known
function of the candidate's content:

    select  <=>  the CV evidences at least 70% of the skills the JD requires

Nothing else determines the outcome. The job descriptions state their
requirements explicitly, the CVs evidence skills in ordinary prose, and
distractor skills are included so the task is not solvable by counting words.
There is zero label noise, by construction.

HOW TO READ THE RESULT

  >90%    the pipeline works, and the mid-50s on the real dataset is the data's
          ceiling, not our bug. This is the sentence the results chapter needs.
  78-90%  partial: relational reasoning is happening, but short of a task whose
          answer is stated in the inputs with no label noise.
  <=78%   at or below what a bag of words manages without understanding the
          relation at all. That is a defect in the system, not in the dataset,
          and it changes what this project concludes.

TWO REFERENCE LINES, MEASURED ON THE SAME CASES

  oracle        logistic regression on the skill-overlap count: 100%. Confirms
                the label really is a deterministic function of the content.
  bag of words  TF-IDF + logistic regression on raw text: ~78%. Not a generator
                fault — the rule is RELATIONAL (which skills appear in BOTH
                documents) and a bag of words cannot represent that.

The 78% line is the one that matters. Comparing two documents and reasoning
about their overlap is precisely the capability an LLM is supposed to add over
lexical matching, so this is where that advantage should appear if it exists
anywhere.

Usage:
    python src/sanity_benchmark.py --build --n 500
    python src/sanity_benchmark.py --eval --n 100
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import random
import argparse
from pathlib import Path

CASES_PATH = Path("data/processed/sanity_cases.jsonl")
RESULTS_PATH = Path("data/processed/eval_sanity.json")
VOCAB_PATH = Path("data/vocab/skills.txt")

N_REQUIRED = 5
THRESHOLD = 0.7          # select iff coverage >= 70% of required skills
N_DISTRACTORS = (2, 5)   # extra CV skills that are NOT required

ROLES = [
    "data scientist", "backend engineer", "devops engineer", "data analyst",
    "machine learning engineer", "frontend developer", "cloud architect",
    "product manager", "qa engineer", "database administrator",
]

JD_TEMPLATE = """we are hiring a {role} to join our engineering team.

required skills: {required}.

the successful candidate will apply these skills daily, own delivery end to end,
and work closely with product and engineering partners. we expect demonstrated
hands-on experience with each of the required skills listed above."""

CV_HEADER = """{name}
{role}

professional summary:
experienced {role} with {years} years of industry experience delivering
production systems in cross-functional teams.

professional experience:
"""

BULLETS = [
    "built and maintained production services using {skill}, owning design through deployment",
    "led a team project that relied heavily on {skill} to meet delivery targets",
    "applied {skill} across several client engagements, including hands-on implementation",
    "improved system reliability by introducing {skill} into the delivery pipeline",
    "mentored two junior engineers in {skill} and ran internal training on it",
]

FIRST = ["alex", "jordan", "sam", "casey", "riley", "morgan", "taylor", "jamie"]
LAST = ["carter", "reed", "hayes", "brooks", "quinn", "ellis", "shaw", "novak"]


def load_vocab():
    return [s.strip().lower() for s in
            VOCAB_PATH.read_text(encoding="utf-8").splitlines() if s.strip()]


def build(n, seed):
    rng = random.Random(seed)
    vocab = load_vocab()
    if len(vocab) < N_REQUIRED + N_DISTRACTORS[1]:
        raise SystemExit("skills vocabulary too small to generate cases")

    # Coverage levels: 0-3 of 5 are reject (0-60%), 4-5 are select (80-100%).
    # Sampled in equal numbers per class so the set is balanced by construction.
    reject_levels, select_levels = [0, 1, 2, 3], [4, 5]

    cases = []
    for i in range(n):
        want_select = i % 2 == 0
        covered_n = rng.choice(select_levels if want_select else reject_levels)

        role = rng.choice(ROLES)
        required = rng.sample(vocab, N_REQUIRED)
        covered = rng.sample(required, covered_n)
        distractors = rng.sample([s for s in vocab if s not in required],
                                 rng.randint(*N_DISTRACTORS))
        cv_skills = covered + distractors
        rng.shuffle(cv_skills)

        coverage = covered_n / N_REQUIRED
        decision = "select" if coverage >= THRESHOLD else "reject"
        assert decision == ("select" if want_select else "reject")

        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        cv = CV_HEADER.format(name=name, role=role, years=rng.randint(3, 12))
        for s in cv_skills:
            cv += "- " + rng.choice(BULLETS).format(skill=s) + "\n"
        cv += ("\neducation:\nbsc computer science\n\nskills: "
               + ", ".join(cv_skills) + "\n")

        cases.append({
            "case_id": f"sanity_{i:05d}",
            "role": role,
            "cv_text": cv,
            "jd_text": JD_TEMPLATE.format(role=role, required=", ".join(required)),
            "decision": decision,
            "decision_reason": (f"evidences {covered_n} of {N_REQUIRED} required "
                                f"skills ({coverage:.0%})"),
            "metadata": {"source": "sanity_benchmark", "coverage": coverage,
                         "n_covered": covered_n, "n_required": N_REQUIRED,
                         "covered": sorted(covered),
                         "missing": sorted(set(required) - set(covered)),
                         "distractors": sorted(distractors)},
            "entities": {"skills_cv": sorted(cv_skills),
                         "skills_jd": sorted(required),
                         "degrees": ["bsc"], "certifications": []},
        })

    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CASES_PATH, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    n_sel = sum(1 for c in cases if c["decision"] == "select")
    print(f"Wrote {len(cases)} cases to {CASES_PATH}")
    print(f"  select {n_sel} / reject {len(cases)-n_sel}")
    print(f"  rule: select iff coverage >= {THRESHOLD:.0%} of {N_REQUIRED} "
          f"required skills")
    print(f"  distractor skills per CV: {N_DISTRACTORS[0]}-{N_DISTRACTORS[1]}")
    return cases


def reference_check(cases):
    """Two references, measuring different things.

    oracle — logistic regression on the skill-overlap count alone. Must be
        100%: it confirms the label really is a deterministic function of the
        candidate's content, i.e. the generator did what it claims.

    bag_of_words — TF-IDF + logistic regression on the raw text. Caps around
        78%, and that is NOT a generator fault. The rule is RELATIONAL — it
        depends on which skills appear in BOTH documents — and a bag of words
        cannot represent that: the same token in the CV and in the JD is
        indistinguishable from the token appearing in only one.

    That gap is what makes this benchmark discriminating rather than merely
    easy. Comparing two documents and reasoning about their overlap is exactly
    the capability an LLM is supposed to add over lexical matching. If the LLM
    cannot clear the 78% bag-of-words line on a task with explicit
    requirements, unambiguous evidence and zero label noise, then it is not
    doing the relational reasoning the whole pipeline assumes it does.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline

    y = np.array([1 if c["decision"] == "select" else 0 for c in cases])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    overlap = np.array([[len(set(c["entities"]["skills_jd"]) &
                             set(c["entities"]["skills_cv"]))] for c in cases])
    oracle = cross_val_score(LogisticRegression(), overlap, y, cv=cv,
                             scoring="accuracy").mean()

    text = [c["cv_text"] + " " + c["jd_text"] for c in cases]
    pipe = make_pipeline(
        TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=2000))
    bow = cross_val_score(pipe, text, y, cv=cv, scoring="accuracy",
                          n_jobs=-1).mean()
    return {"oracle": float(oracle), "bag_of_words": float(bow)}


def evaluate(n, seed):
    import step5_llm_decision as s5
    from exp_prompt_variants import c_zeroshot
    from step6_evaluate import compute_metrics

    cases = [json.loads(l) for l in open(CASES_PATH, encoding="utf-8")]
    rng = random.Random(seed)
    sel = [c for c in cases if c["decision"] == "select"]
    rej = [c for c in cases if c["decision"] == "reject"]
    test = rng.sample(sel, min(n // 2, len(sel))) + rng.sample(rej, min(n // 2, len(rej)))
    rng.shuffle(test)

    print(f"Reference check (5-fold on all {len(cases)} cases)...", flush=True)
    ref = reference_check(cases)
    print(f"  oracle (skill-overlap count) : {ref['oracle']:.1%}"
          f"{'  OK — the label is a function of the content'
             if ref['oracle'] > 0.99 else '  <-- GENERATOR PROBLEM'}")
    print(f"  bag of words (TF-IDF+LogReg) : {ref['bag_of_words']:.1%}"
          f"  <- the line the LLM must beat\n")

    rows = []
    for i, c in enumerate(test, 1):
        out = s5.parse_decision(s5.call_llm(c_zeroshot(
            c["role"], c["cv_text"], c["jd_text"],
            c["entities"]["skills_cv"], c["entities"]["skills_jd"], [])))
        pred = str(out.get("decision", "unknown")).lower()
        rows.append({"case_id": c["case_id"], "role": c["role"],
                     "ground_truth": c["decision"], "prediction": pred,
                     "correct": pred == c["decision"],
                     "coverage": c["metadata"]["coverage"],
                     "n_covered": c["metadata"]["n_covered"],
                     "confidence": out.get("confidence", "unknown"),
                     "parse_error": bool(out.get("parse_error"))})
        print(f"  [{i}/{len(test)}] cov {c['metadata']['n_covered']}/5 "
              f"{c['decision'][:3]}->{pred[:3]} "
              f"{'ok' if pred == c['decision'] else 'X'}", flush=True)

    valid = [r for r in rows if r["prediction"] in ("select", "reject")]
    m = compute_metrics(valid)

    print("\n" + "=" * 72)
    print("SANITY BENCHMARK — the label IS a function of the CV")
    print("=" * 72)
    print(f"  oracle (overlap count)    : {ref['oracle']:.1%}")
    print(f"  bag of words (TF-IDF)     : {ref['bag_of_words']:.1%}")
    print(f"  LLM zero-shot accuracy    : {m['accuracy']:.1%}")
    print(f"  precision / recall / F1   : {m['precision']:.1%} / "
          f"{m['recall']:.1%} / {m['f1']:.1%}")
    print(f"  select rate               : {m['pred_select_rate']:.1%}")
    print()
    print("  accuracy by skill coverage:")
    for k in sorted({r["n_covered"] for r in rows}):
        g = [r for r in rows if r["n_covered"] == k]
        acc = sum(r["correct"] for r in g) / len(g)
        selr = sum(1 for r in g if r["prediction"] == "select") / len(g)
        print(f"    {k}/5 covered ({k/5:.0%}, truth "
              f"{'select' if k/5 >= THRESHOLD else 'reject'}): "
              f"acc {acc:.0%}  said-select {selr:.0%}  (n={len(g)})")
    print()
    if m["accuracy"] > 0.9:
        print("  >90%: the pipeline works. The mid-50s on the real dataset is")
        print("  the data's ceiling, not a defect in the system.")
    elif m["accuracy"] > ref["bag_of_words"]:
        print(f"  Beats the {ref['bag_of_words']:.0%} bag-of-words line, so some")
        print("  relational reasoning is happening — but well short of a task")
        print("  whose answer is stated in the inputs with zero label noise.")
    else:
        print(f"  At or below the {ref['bag_of_words']:.0%} bag-of-words line, on a")
        print("  task with explicit requirements, unambiguous evidence and zero")
        print("  label noise. The LLM is not doing the relational reasoning the")
        print("  pipeline assumes it does. That is a defect in the system, not")
        print("  in the dataset, and it changes what this project concludes.")
    print("=" * 72)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"config": {"n": len(test), "seed": seed,
                              "model": s5.MODEL_NAME,
                              "temperature": s5.TEMPERATURE,
                              "threshold": THRESHOLD, "n_required": N_REQUIRED,
                              "evidence": "zero-shot"},
                   "reference": {k: round(v, 4) for k, v in ref.items()},
                   "metrics": m, "results": rows}, f, indent=2)
    tmp.replace(RESULTS_PATH)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    if a.build:
        build(a.n, a.seed)
    if a.eval:
        if not CASES_PATH.exists():
            raise SystemExit("build the benchmark first: --build --n 500")
        evaluate(a.n, a.seed)
    if not (a.build or a.eval):
        p.error("pass --build and/or --eval")
