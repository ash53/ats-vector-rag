"""
Regenerate RESULTS.md from the raw result files.

Every number in the write-up should come from here rather than being copied by
hand out of a terminal. Transcription is where numbers quietly rot: a figure
gets pasted into a document, the experiment is re-run, and the document keeps
the old value with no way to tell. This reads data/processed/eval_*.json and
emits the tables, so re-running it after any experiment updates everything at
once and missing files are reported as pending rather than silently omitted.

Usage:
    python src/make_results_table.py            # writes RESULTS.md
    python src/make_results_table.py --stdout
"""

import json
import argparse
from pathlib import Path

PROC = Path("data/processed")
OUT = Path("RESULTS.md")


def load(name):
    p = PROC / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"_error": f"{name} is not valid JSON: {e}"}


def pct(x):
    return f"{x:.1%}" if isinstance(x, (int, float)) else "—"


def row(label, m, extra=""):
    return (f"| {label} | {pct(m.get('accuracy'))} | {pct(m.get('precision'))} | "
            f"{pct(m.get('recall'))} | {pct(m.get('f1'))} | "
            f"{pct(m.get('pred_select_rate'))} | {m.get('total_cases','—')} |{extra}")


def section_decision(lines):
    lines.append("## 1. Decision quality\n")
    lines.append("All runs seed 42, `llama3.1:8b`, leak-free index.\n")
    lines.append("| System | Accuracy | Precision | Recall | F1 | Select rate | n |")
    lines.append("|---|---|---|---|---|---|---|")

    rag300 = load("eval_results_n300.json")
    if rag300:
        lines.append(row("RAG pipeline (two-view, temp 0.8)", rag300["metrics"]))

    base = load("eval_baselines_n300.json")
    if base:
        names = {"always-select": "always-select (degenerate)",
                 "keyword-tfidf": "Keyword TF-IDF (traditional ATS)",
                 "supervised-logreg": "Supervised TF-IDF+LogReg (ceiling)"}
        for k, v in base["baselines"].items():
            lines.append(row(names.get(k, k), v["metrics"]))

    rag40 = load("eval_results_n40.json")
    if rag40:
        lines.append(row("RAG pipeline (n=40, temp 0.8)", rag40["metrics"]))
    lines.append("")
    lines.append("A balanced test set makes *always-select* score 50.0% accuracy "
                 "and 66.7% F1 without doing anything, so no F1 here is "
                 "interpretable without the select rate beside it.\n")


def section_controls(lines):
    d = load("eval_controls_n300.json")
    lines.append("## 2. Does retrieval contribute? (controls)\n")
    if not d:
        lines.append("_Pending — run `exp_prompt_variants.py --variants c_*`._\n")
        return
    lines.append("Prompt held fixed, evidence varied, same 300 cases, "
                 f"temperature {d['config'].get('temperature', '?')}.\n")
    lines.append("| Condition | Accuracy | Precision | Recall | F1 | Select rate | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in d["variants"].items():
        lines.append(row(f"`{k}` — {v['label']}", v["metrics"]))
    lines.append("")


def section_ablations(lines):
    d = load("eval_ablations_n300.json")
    lines.append("## 3. Does the model read the CV? (input ablations)\n")
    if not d:
        lines.append("_Pending — run `exp_prompt_variants.py --variants a_*`._\n")
        return
    lines.append("Zero-shot throughout; only the candidate information changes. "
                 "Compare against `c_zeroshot` in section 2.\n")
    lines.append("| Condition | Accuracy | Precision | Recall | F1 | Select rate | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in d["variants"].items():
        lines.append(row(f"`{k}` — {v['label']}", v["metrics"]))
    lines.append("")


def section_prompts(lines):
    d = load("eval_prompt_variants.json")
    lines.append("## 4. Prompt variants\n")
    if not d:
        lines.append("_Pending._\n")
        return
    temp = d["config"].get("temperature", "0.8 (unpinned)")
    lines.append(f"n={d['config']['n']}, temperature {temp}.")
    if d["config"]["n"] < 100:
        lines.append("\n> **Ranking not trustworthy at this sample size.** The "
                     "identical prompt on the identical cases scored 45.0% and "
                     "then 50.0% before temperature was pinned. Treat the "
                     "mechanical findings as the result, not the ordering.\n")
    lines.append("| Variant | Accuracy | Precision | Recall | F1 | Select rate | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in d["variants"].items():
        lines.append(row(f"`{k}` — {v['label']}", v["metrics"]))
    lines.append("")


def section_retrieval(lines):
    d = load("eval_retrieval_views.json")
    lines.append("## 5. Retrieval layer (no LLM)\n")
    if not d:
        lines.append("_Pending — run `exp_retrieval_views.py`._\n")
        return
    c = d["config"]
    lines.append(f"{c['n']} queries, leave-one-out, k={c['k']}.\n")
    lines.append("| View | same-decision@k | same-role@k | skill Jaccard | "
                 "similarity spread | exemplar diversity |")
    lines.append("|---|---|---|---|---|---|")
    for k, v in d["views"].items():
        lines.append(f"| {k} | {pct(v['same_decision_at_k'])} | "
                     f"{pct(v['same_role_at_k'])} | {v['skill_jaccard']:.3f} | "
                     f"{v['similarity_spread']:.4f} | "
                     f"{v['exemplar_diversity']:.4f} |")
    lines.append("")
    lines.append("same-decision@k at 50% would mean the retrieved outcomes carry "
                 "no information about the query. This is the ceiling on what "
                 "*any* retrieval strategy can supply here.\n")


def section_counterfactual(lines):
    lines.append("## 6. Counterfactual sensitivity\n")
    files = sorted(PROC.glob("eval_counterfactual_*.json"))
    if not files:
        lines.append("_Pending — run `exp_counterfactual.py --mode counterfactual`._\n")
        return
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    lines.append("Does the decision move the way the evidence demands?\n")
    lines.append("| Arm | Unchanged | Expected | Contradictory | Directionally right |")
    lines.append("|---|---|---|---|---|")
    for arm in ["inject", "remove"]:
        rows = [r for r in d["results"] if r["arm"] == arm]
        if not rows:
            continue
        c = {v: sum(1 for r in rows if r["verdict"] == v)
             for v in ["unchanged", "expected", "contradictory"]}
        moved = c["expected"] + c["contradictory"]
        share = f"{c['expected']/moved:.0%}" if moved else "—"
        lines.append(f"| {arm} (n={len(rows)}) | {c['unchanged']} | "
                     f"{c['expected']} | {c['contradictory']} | {share} |")
    lines.append("")
    lines.append("'Directionally right' counts only the decisions that moved; "
                 "50% there is a coin flip.\n")


def section_fairness(lines):
    lines.append("## 7. Fairness — matched-pair name swap\n")
    files = sorted(PROC.glob("eval_fairness_*.json"))
    if not files:
        lines.append("_Pending — run `exp_counterfactual.py --mode fairness`._\n")
        return
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    rows = d["results"]
    groups = sorted({r["group"] for r in rows})
    lines.append("Identical CVs; only the candidate's name differs "
                 "(Bertrand & Mullainathan 2004 design).\n")
    lines.append("| Group | Select rate | n |")
    lines.append("|---|---|---|")
    rates = {}
    for g in groups:
        sel = [r for r in rows if r["group"] == g]
        rates[g] = sum(1 for r in sel if r["decision"] == "select") / len(sel)
        lines.append(f"| {g} | {pct(rates[g])} | {len(sel)} |")
    lines.append("")
    if rates:
        hi, lo = max(rates, key=rates.get), min(rates, key=rates.get)
        ratio = rates[lo] / rates[hi] if rates[hi] else float("nan")
        lines.append(f"Largest gap: **{hi} {pct(rates[hi])} vs {lo} "
                     f"{pct(rates[lo])}** = {rates[hi]-rates[lo]:.1%} points. "
                     f"Impact ratio {ratio:.2f} — "
                     f"{'passes' if ratio >= 0.8 else '**fails**'} the "
                     f"four-fifths rule.\n")
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["group"]] = r["decision"]
    complete = [v for v in by_case.values() if len(v) == len(groups)]
    if complete:
        flipped = sum(1 for v in complete if len(set(v.values())) > 1)
        lines.append(f"Decisions that changed on the name alone: "
                     f"**{flipped}/{len(complete)} = "
                     f"{flipped/len(complete):.1%}**\n")


def section_feedback(lines):
    d = load("eval_feedback.json")
    lines.append("## 8. Rejection feedback — automatic checks\n")
    if not d:
        lines.append("_Pending — run `step7_feedback.py --n 20`._\n")
        return
    s = d["summary"]
    lines.append("| Check | Failures | Rate |")
    lines.append("|---|---|---|")
    t = s["total"]
    for k, lbl in [("ungrounded", "Ungrounded gaps (not a stated requirement)"),
                   ("contradicted", "**Contradicted gaps (the CV disproves them)**"),
                   ("neutrality", "Protected-characteristic or pronoun hits")]:
        v = s.get(k, 0)
        lines.append(f"| {lbl} | {v}/{t} | {v/t:.0%} |")
    lines.append(f"| Passed every check | {s['passed']}/{t} | {s['passed']/t:.0%} |")
    lines.append("")


def main(to_stdout=False):
    lines = ["# Results",
             "",
             "_Generated by `src/make_results_table.py` from "
             "`data/processed/eval_*.json`. Do not edit by hand — re-run it._",
             "",
             "Interpretation, caveats and the reasoning behind these numbers live "
             "in `EVALUATION_AND_APPROACH_PLAN.md`. This file is the numbers only.",
             "",
             "---",
             ""]
    for fn in [section_decision, section_controls, section_ablations,
               section_prompts, section_retrieval, section_counterfactual,
               section_fairness, section_feedback]:
        fn(lines)
        lines.append("---\n")

    text = "\n".join(lines)
    if to_stdout:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        pending = text.count("_Pending")
        print(f"Wrote {OUT} ({len(text.splitlines())} lines, "
              f"{pending} section(s) still pending)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stdout", action="store_true")
    main(to_stdout=p.parse_args().stdout)
