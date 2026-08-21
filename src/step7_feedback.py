"""
STEP 7 — Rejection feedback generator

The proposal requires a structured, constructive, bias-free message for rejected
candidates, in four parts:

  1. Decision context   neutral summary of what the role required
  2. Specific gaps      2-3 concrete gaps, each tied to a stated requirement
  3. Suggestions        actionable next steps — courses, certifications, projects
  4. Encouragement      a genuine invitation to re-apply

Step 5 already emits `key_gaps`; this turns them into something a candidate can
actually receive. It runs zero-shot, since the retrieval controls showed
exemplars contribute nothing (eval_controls_n300.json, p=0.503).

WHY THE CHECKS MATTER HERE MORE THAN ANYWHERE ELSE

Everything else in this project is a number in a report. This is text sent to a
person who was turned down for a job. Two failure modes are unacceptable, and
both are measurable, so both are checked automatically on every message:

  groundedness  A gap is only legitimate if it traces to a requirement in the
                job description or to the extracted skill gap. The decision
                labels in this dataset are uncorrelated with candidate data
                (see EVALUATION_AND_APPROACH_PLAN.md §0.2), so a model asked to
                justify one will invent plausible-sounding reasons. Telling
                someone they lack a skill they never claimed, or that the job
                never asked for, is a fabricated accusation about their career.
  contradiction A gap that asserts the candidate lacks something their CV
                demonstrably has. Strictly worse than ungrounded: not merely
                unsupported but false, and false about the person reading it.
                This is not hypothetical — the first two-message batch produced
                one. A devops candidate whose CV lists kubernetes, aws, azure,
                gcp and python was told they had not evidenced containerisation,
                cloud platforms or scripting. Three false claims, none of them
                requirements the job description ever stated.
  neutrality    The message must not mention or imply name, gender, age,
                nationality or any protected characteristic. Checked on word
                boundaries plus a pronoun scan.

A message that fails any check is flagged and must not be sent. Failure rates
are reported per batch — that number belongs in the report.

Usage:
    python src/step7_feedback.py --case-id case_01234
    python src/step7_feedback.py --n 20          # batch, with check statistics
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re
import json
import time
import random
import argparse
from pathlib import Path

import step5_llm_decision as s5
from step6_evaluate import load_cases

CASES_PATH = Path("data/processed/cases_stage2.jsonl")
RESULTS_PATH = Path("data/processed/eval_feedback.json")

# Terms that must never appear in a candidate-facing rejection message.
# Matched on WORD BOUNDARIES, not as substrings: an earlier version flagged
# "database management" for containing "man" and "age", which would have marked
# almost every legitimate message as biased. A filter that fires on everything
# is no more useful than no filter.
PROTECTED_WORDS = [
    "male", "female", "man", "men", "woman", "women", "gender", "age", "aged",
    "young", "younger", "old", "older", "elderly", "race", "racial",
    "nationality", "foreign", "foreigner", "native", "accent", "married",
    "marriage", "spouse", "family", "children", "kids", "maternity", "visa",
]
# Prefixes, where any word starting with them is a hit.
PROTECTED_PREFIXES = ["pregnan", "disab", "religio", "ethnic"]
PROTECTED_PHRASES = ["cultural fit", "culture fit", "team fit"]
GENDERED_PRONOUNS = ["he", "she", "his", "her", "him", "hers", "himself", "herself"]


def build_feedback_prompt(role, jd_text, skills_jd, missing_skills, key_gaps):
    gaps = ", ".join(key_gaps) if key_gaps else ", ".join(missing_skills) or "none identified"
    return f"""You are writing feedback to a candidate who applied for a role and was not selected.
The message will be sent to them directly. It must be honest, specific, respectful and useful.

## THE ROLE
- Title: {role}
- Job description: {jd_text}
- Required skills as stated: {", ".join(skills_jd) or "none explicitly listed"}

## WHAT THE ASSESSMENT FOUND
- Skills required but not evidenced in the application: {", ".join(missing_skills) or "none"}
- Gaps noted by the assessment: {gaps}

## RULES — these are not style preferences
1. Only mention a gap if it appears in the lists above. Do NOT invent gaps, and
   do NOT infer anything about the candidate beyond what is listed. If the lists
   are empty, say the decision was close and based on the strength of the field.
2. Never mention or imply gender, age, race, nationality, family circumstances,
   health, religion, or "cultural fit". Never use he/she/his/her — write in the
   second person, addressing the candidate as "you".
3. Do not speculate about the candidate's character, motivation or potential.
4. Be concrete. "Build a portfolio project using X" beats "gain more experience".
5. No false hope and no false praise. Do not promise a future outcome.

## WRITE FOUR PARTS
1. context: 1-2 sentences on what the role required. Neutral, no verdict language.
2. gaps: 2-3 specific gaps, each tied to a stated requirement.
3. suggestions: 2-3 concrete, actionable next steps matched to those gaps.
4. encouragement: 1-2 sentences, genuine, inviting a future application.

Respond in this exact JSON format only - no extra text:
{{
    "context": "...",
    "gaps": ["...", "..."],
    "suggestions": ["...", "..."],
    "encouragement": "..."
}}

JSON Response:"""


def check_groundedness(feedback, skills_jd, missing_skills, jd_text):
    """Every cited gap must trace to a stated requirement or an extracted gap."""
    allowed = {s.lower() for s in list(skills_jd) + list(missing_skills)}
    jd_words = set(re.findall(r"[a-z]{4,}", jd_text.lower()))
    ungrounded = []
    for gap in feedback.get("gaps", []):
        g = str(gap).lower()
        if any(skill in g for skill in allowed):
            continue
        # allow a gap phrased with words the JD itself uses
        if allowed == set() and jd_words & set(re.findall(r"[a-z]{4,}", g)):
            continue
        ungrounded.append(gap)
    return ungrounded


def _load_vocab():
    path = Path("data/vocab/skills.txt")
    return [s.strip().lower() for s in path.read_text(encoding="utf-8").splitlines()
            if s.strip()] if path.exists() else []


_VOCAB = _load_vocab()


def check_contradiction(feedback, case):
    """Gaps that assert the candidate lacks something their CV demonstrably has.

    Strictly worse than an ungrounded gap: not merely unsupported, but false,
    and false about the person receiving the message. Observed in the very first
    batch — a devops candidate whose CV lists kubernetes, aws, azure, gcp and
    python was told they had not evidenced containerisation, cloud platforms or
    scripting. All three claims were contradicted by their own application.
    """
    have = {s.lower() for s in case["entities"]["skills_cv"]}
    cv_lower = case["cv_text"].lower()
    contradicted = []
    for gap in feedback.get("gaps", []):
        g = str(gap).lower()
        # only treat it as an absence claim if it is phrased as one
        if not re.search(r"\b(no|not|didn't|did not|lack|lacks|lacking|missing|"
                         r"absent|unable to find|without|little|limited)\b", g):
            continue
        for skill in _VOCAB:
            if re.search(rf"\b{re.escape(skill)}\b", g) and (
                    skill in have or re.search(rf"\b{re.escape(skill)}\b", cv_lower)):
                contradicted.append({"gap": gap, "skill_actually_present": skill})
                break
    return contradicted


def check_neutrality(feedback):
    """Flag protected characteristics and gendered pronouns, on word boundaries."""
    text = " ".join([str(feedback.get("context", "")),
                     " ".join(map(str, feedback.get("gaps", []))),
                     " ".join(map(str, feedback.get("suggestions", []))),
                     str(feedback.get("encouragement", ""))]).lower()
    words = set(re.findall(r"[a-z']+", text))

    hits = sorted(words & set(PROTECTED_WORDS))
    hits += sorted(w for w in words if any(w.startswith(p) for p in PROTECTED_PREFIXES))
    hits += [p for p in PROTECTED_PHRASES if p in text]
    hits += sorted(words & set(GENDERED_PRONOUNS))
    return sorted(set(hits))


def render(feedback):
    return (f"{feedback.get('context','').strip()}\n\n"
            "Where your application fell short:\n"
            + "".join(f"  - {g}\n" for g in feedback.get("gaps", []))
            + "\nWhat would strengthen a future application:\n"
            + "".join(f"  - {s}\n" for s in feedback.get("suggestions", []))
            + f"\n{feedback.get('encouragement','').strip()}\n")


def generate(case):
    missing = sorted(set(case["entities"]["skills_jd"]) - set(case["entities"]["skills_cv"]))

    # Reuse Step 5's decision to source key_gaps, zero-shot (no exemplars).
    from exp_prompt_variants import c_zeroshot
    decision = s5.parse_decision(s5.call_llm(c_zeroshot(
        case["role"], case["cv_text"], case["jd_text"],
        case["entities"]["skills_cv"], case["entities"]["skills_jd"], [])))

    prompt = build_feedback_prompt(case["role"], case["jd_text"],
                                   case["entities"]["skills_jd"], missing,
                                   decision.get("key_gaps", []))
    feedback = s5.parse_decision(s5.call_llm(prompt))  # same JSON extractor

    ungrounded = check_groundedness(feedback, case["entities"]["skills_jd"],
                                    missing, case["jd_text"])
    contradicted = check_contradiction(feedback, case)
    neutrality = check_neutrality(feedback)
    return {
        "case_id": case["case_id"],
        "role": case["role"],
        "ground_truth": case["decision"].lower(),
        "predicted_decision": str(decision.get("decision", "unknown")).lower(),
        "missing_skills": missing,
        "key_gaps": decision.get("key_gaps", []),
        "feedback": feedback,
        "ungrounded_gaps": ungrounded,
        "contradicted_gaps": contradicted,
        "neutrality_hits": neutrality,
        "passes_checks": not ungrounded and not contradicted and not neutrality,
        "parse_error": bool(feedback.get("parse_error")),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", type=str, default=None)
    p.add_argument("--n", type=int, default=None, help="batch size (rejected cases)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(RESULTS_PATH))
    a = p.parse_args()

    cases = load_cases(CASES_PATH)
    by_id = {c["case_id"]: c for c in cases}

    if a.case_id:
        targets = [by_id[a.case_id]]
    else:
        rejected = [c for c in cases if c["decision"].lower() == "reject"]
        random.Random(a.seed).shuffle(rejected)
        targets = rejected[:a.n or 5]

    rows, t0 = [], time.time()
    for i, c in enumerate(targets, 1):
        r = generate(c)
        rows.append(r)
        flag = "ok" if r["passes_checks"] else "FLAGGED"
        print(f"[{i}/{len(targets)}] {c['case_id']} {c['role'][:22]:22s} {flag}",
              flush=True)
        if not r["passes_checks"]:
            if r["ungrounded_gaps"]:
                print(f"    ungrounded gaps: {r['ungrounded_gaps']}")
            for cg in r["contradicted_gaps"]:
                print(f"    CONTRADICTED — claims a gap in "
                      f"'{cg['skill_actually_present']}', which the CV has")
            if r["neutrality_hits"]:
                print(f"    neutrality hits: {r['neutrality_hits']}")

    if len(targets) == 1:
        print("\n" + "=" * 72)
        print(render(rows[0]["feedback"]))
        print("=" * 72)

    passed = sum(1 for r in rows if r["passes_checks"])
    ung = sum(1 for r in rows if r["ungrounded_gaps"])
    con = sum(1 for r in rows if r["contradicted_gaps"])
    neu = sum(1 for r in rows if r["neutrality_hits"])
    print(f"\npassed both checks : {passed}/{len(rows)} = {passed/len(rows):.0%}")
    print(f"ungrounded gaps    : {ung}/{len(rows)}")
    print(f"CONTRADICTED gaps  : {con}/{len(rows)}   "
          f"(claims a gap the CV disproves)")
    print(f"neutrality flags   : {neu}/{len(rows)}")
    print(f"elapsed            : {time.time()-t0:.0f}s")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"config": {"n": len(rows), "seed": a.seed,
                              "model": s5.MODEL_NAME,
                              "temperature": s5.TEMPERATURE,
                              "evidence": "zero-shot"},
                   "summary": {"passed": passed, "ungrounded": ung,
                               "contradicted": con, "neutrality": neu,
                               "total": len(rows)},
                   "results": rows}, f, indent=2)
    tmp.replace(out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
