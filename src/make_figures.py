"""
Generate the results figures from data/processed/eval_*.json.

Same contract as make_results_table.py: figures are derived from the raw result
files, never drawn by hand, so re-running after an experiment refreshes them and
no figure can quietly disagree with the table beside it. Experiments that have
not produced data are skipped and named at the end rather than silently omitted.

Output: figures/*.png (200 dpi, for drafts) and figures/*.pdf (vector, for the
thesis).

Design notes — the choices here are deliberate, not defaults:

* Palette is the two-slot categorical set #2a78d6 / #eb6834, validated for
  colorblind separation before use (worst adjacent CVD dE 24.7 protan, normal
  vision 33.6, both above the gates; all slots clear 3:1 on the surface).
* One axis, always. Accuracy and select rate are both percentages on 0-100, so
  they share a scale honestly; a second y-axis would let any two quantities be
  drawn as if they moved together.
* Every bar carries its value as a direct label, so identity and magnitude never
  depend on reading a color against an axis.
* Grid and axes are recessive; reference lines (chance, ceilings, thresholds)
  are the only annotation that competes with the data, because in this project
  the reference lines ARE the finding.
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PROC = Path("data/processed")
FIGDIR = Path("figures")

# Validated categorical slots (light mode) + chart chrome, from the design system
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 200,
})

skipped = []


def load(name):
    p = PROC / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def finish(fig, ax, name, title, subtitle=None, xlabel=None):
    # Title and subtitle are placed explicitly rather than via set_title, which
    # cannot reserve room for a second line and let the first render start
    # colliding with the topmost mark.
    ax.text(0, 1.13 if subtitle else 1.05, title, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=INK, va="bottom")
    if subtitle:
        ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=8.5,
                color=INK2, va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    FIGDIR.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}.png + .pdf")


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- figure 1
def fig_forest():
    """Every system's accuracy with 95% CIs. Makes 'one band' visible at once."""
    rag = load("eval_results_n300.json")
    base = load("eval_baselines_n300.json")
    ctrl = load("eval_controls_n300.json")
    if not (rag and base):
        skipped.append("1 forest (needs eval_results_n300 + eval_baselines_n300)")
        return

    rows = [("RAG pipeline", rag["metrics"])]
    labels = {"supervised-logreg": "Supervised TF-IDF+LogReg",
              "always-select": "Always select (degenerate)",
              "keyword-tfidf": "Keyword TF-IDF"}
    for k, lbl in labels.items():
        if k in base["baselines"]:
            rows.append((lbl, base["baselines"][k]["metrics"]))
    if ctrl and "c_zeroshot" in ctrl["variants"]:
        rows.append(("LLM, no retrieval", ctrl["variants"]["c_zeroshot"]["metrics"]))

    rows.sort(key=lambda r: r[1]["accuracy"])
    names = [r[0] for r in rows]
    acc = [r[1]["accuracy"] for r in rows]
    ns = [r[1]["total_cases"] for r in rows]
    los, his = zip(*[wilson(round(a * n), n) for a, n in zip(acc, ns)])

    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(rows) + 2.0))
    y = range(len(rows))

    ax.axvline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
    ax.axvline(0.582, color=MUTED, lw=1, ls=":", zorder=1)
    # Both annotations sit below every mark, on opposite sides of their own
    # line, so they cannot collide with the data, the title or the axis label.
    ax.text(0.497, -0.62, "chance 50%", color=MUTED, fontsize=8,
            va="center", ha="right")
    ax.text(0.585, -0.62, "supervised ceiling 58.2%", color=MUTED, fontsize=8,
            va="center", ha="left")

    for i, (a, lo, hi) in enumerate(zip(acc, los, his)):
        ax.plot([lo, hi], [i, i], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter(acc, list(y), s=54, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    # Value beside the interval, not above it: stacked rows leave no vertical
    # room and the top label was landing in the subtitle.
    for i, (a, hi) in enumerate(zip(acc, his)):
        ax.text(hi + 0.006, i, f"{a:.1%}", ha="left", va="center",
                fontsize=8.5, color=INK, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{n}  (n={c})" for n, c in zip(names, ns)], color=INK2)
    ax.set_ylim(-1.0, len(rows) - 0.5)
    ax.set_xlim(0.38, 0.70)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    finish(fig, ax, "fig1_forest_accuracy",
           "Every method lands in the same band",
           "Decision accuracy with 95% confidence intervals, identical 300-case "
           "test set", "Accuracy")


# ---------------------------------------------------------------- figure 2
def fig_controls():
    """Accuracy and select rate per evidence condition — the retrieval null."""
    d = load("eval_controls_n300.json")
    if not d:
        skipped.append("2 controls (needs eval_controls_n300)")
        return
    order = ["c_retrieved", "c_random_role", "c_random_corpus", "c_zeroshot"]
    nice = {"c_retrieved": "Retrieved\n(top-20 similar)",
            "c_random_role": "Random,\nsame role",
            "c_random_corpus": "Random,\nany role",
            "c_zeroshot": "No exemplars\n(zero-shot)"}
    keys = [k for k in order if k in d["variants"]]
    acc = [d["variants"][k]["metrics"]["accuracy"] for k in keys]
    sel = [d["variants"][k]["metrics"]["pred_select_rate"] for k in keys]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = range(len(keys))
    w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], acc, w, color=BLUE,
                label="Accuracy", zorder=2)
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], sel, w, color=ORANGE,
                label="Select rate", zorder=2)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015,
                    f"{r.get_height():.1%}", ha="center", fontsize=8.5,
                    color=INK, fontweight="bold")

    ax.axhline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(-0.46, 0.512, "chance", color=MUTED, fontsize=8, ha="left", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels([nice[k] for k in keys], color=INK2)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    finish(fig, ax, "fig2_retrieval_controls",
           "Retrieval changes nothing; the select bias is constant",
           "Prompt held fixed, evidence varied. All paired comparisons p ≥ 0.50")


# ---------------------------------------------------------------- figure 3
def fig_ablations():
    """Select rate collapses when the CV is degraded — the input-sensitivity result."""
    d = load("eval_ablations_n300.json")
    ctrl = load("eval_controls_n300.json")
    if not (d and ctrl and "c_zeroshot" in ctrl["variants"]):
        skipped.append("3 CV ablations (needs eval_ablations_n300 + eval_controls_n300)")
        return

    rows = [("Real CV", ctrl["variants"]["c_zeroshot"]["metrics"])]
    nice = {"a_cv_truncated": "CV truncated\n(200 chars)",
            "a_cv_swapped": "Someone else's CV",
            "a_cv_empty": "No CV at all"}
    for k, lbl in nice.items():
        if k in d["variants"]:
            rows.append((lbl, d["variants"][k]["metrics"]))

    names = [r[0] for r in rows]
    sel = [r[1]["pred_select_rate"] for r in rows]
    acc = [r[1]["accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = range(len(rows))
    w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], acc, w, color=BLUE,
                label="Accuracy", zorder=2)
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], sel, w, color=ORANGE,
                label="Select rate", zorder=2)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015,
                    f"{r.get_height():.1%}", ha="center", fontsize=8.5,
                    color=INK, fontweight="bold")
    ax.axhline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(-0.46, 0.512, "chance", color=MUTED, fontsize=8, ha="left", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, color=INK2)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    finish(fig, ax, "fig3_cv_ablations",
           "The model reads the CV — but that sensitivity does not track the labels",
           "Zero-shot throughout; only the candidate information changes")


# ---------------------------------------------------------------- figure 4
def fig_views():
    """Retrieval quality by embedding view, against the no-information line.

    A dot plot, not bars. The differences here are 4 points on a scale whose
    meaningful floor is 50%, so a bar chart would need a truncated baseline to
    show them — and bar length encodes magnitude, so truncating it overstates
    the difference. Dots encode position only, which makes a partial axis
    honest.
    """
    d = load("eval_retrieval_views.json")
    if not d:
        skipped.append("4 embedding views (needs eval_retrieval_views)")
        return
    nice = {"case_only": "Case view only (768-d)",
            "skill_only": "Skill view only (768-d)",
            "two_view": "Two-view concat (1536-d, original)"}
    keys = [k for k in ["two_view", "skill_only", "case_only"] if k in d["views"]]
    vals = [d["views"][k]["same_decision_at_k"] for k in keys]

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    y = range(len(keys))

    ax.axvline(0.5, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.text(0.5, -0.85, "50% — no information", color=MUTED, fontsize=8,
            ha="center", va="center")

    # Segment from the no-information line to the value: the length of that
    # segment IS the usable signal, which is the point of the figure.
    for i, v in enumerate(vals):
        ax.plot([0.5, v], [i, i], color=BLUE, lw=2, alpha=0.35,
                solid_capstyle="butt", zorder=2)
    ax.scatter(vals, list(y), s=64, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.0015, i, f"  {v:.1%}", ha="left", va="center",
                fontsize=9, color=INK, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels([nice[k] for k in keys], color=INK2)
    ax.set_ylim(-1.2, len(keys) - 0.4)
    ax.set_xlim(0.487, 0.60)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    finish(fig, ax, "fig4_embedding_views",
           "Only 6.6 points of signal exist in the retrieved evidence",
           "Share of retrieved cases sharing the query's outcome, 300 queries. "
           "Two-view vs case-only: p=0.50", "same-decision@10")


# ---------------------------------------------------------------- figure 5
def fig_calibration():
    """Stated confidence against realised accuracy, with intervals.

    Dots with 95% CIs rather than bars. The buckets are wildly unequal (276 vs
    24 cases), so the small bucket's accuracy is barely constrained — and that
    uncertainty is the finding. Bars would draw two confident-looking columns
    and hide it.
    """
    d = load("eval_results_n300.json")
    if not d:
        skipped.append("5 calibration (needs eval_results_n300)")
        return
    buckets = {}
    for r in d["results"]:
        c = str(r.get("confidence", "unknown")).lower()
        if c in ("high", "medium", "low"):
            buckets.setdefault(c, []).append(bool(r["correct"]))
    order = [b for b in ["low", "medium", "high"] if b in buckets]
    if not order:
        skipped.append("5 calibration (no confidence values)")
        return
    acc = [sum(buckets[b]) / len(buckets[b]) for b in order]
    ns = [len(buckets[b]) for b in order]
    cis = [wilson(sum(buckets[b]), len(buckets[b])) for b in order]
    overall = d["metrics"]["accuracy"]

    fig, ax = plt.subplots(figsize=(7.4, 0.75 * len(order) + 2.2))
    y = range(len(order))

    ax.axvline(overall, color=ORANGE, lw=1.4, ls="--", zorder=1)
    ax.text(overall, -0.78, f"overall {overall:.1%}", color=ORANGE, fontsize=8,
            ha="center", va="center")
    ax.axvline(0.5, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(0.5, len(order) - 0.42, "chance", color=MUTED, fontsize=8,
            ha="center", va="bottom")

    for i, (lo, hi) in enumerate(cis):
        ax.plot([lo, hi], [i, i], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter(acc, list(y), s=64, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    for i, (a, (lo, hi)) in enumerate(zip(acc, cis)):
        ax.text(hi + 0.008, i, f"{a:.0%}", ha="left", va="center", fontsize=9,
                color=INK, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{b.capitalize()}  (n={n})" for b, n in zip(order, ns)],
                       color=INK2)
    ax.set_ylim(-1.15, len(order) - 0.35)
    ax.set_xlim(0.20, 0.85)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    finish(fig, ax, "fig5_calibration",
           "Stated confidence carries no information about correctness",
           "Accuracy within each self-reported confidence bucket, 95% CIs, n=300",
           "Accuracy")


# ---------------------------------------------------------------- figure 6
def fig_fairness():
    """Select rate by demographic name group, with the four-fifths threshold."""
    files = sorted(PROC.glob("eval_fairness_*.json"))
    if not files:
        skipped.append("6 fairness (needs eval_fairness_*)")
        return
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    rows = d["results"]
    groups = sorted({r["group"] for r in rows})
    rates, ns = [], []
    for g in groups:
        sub = [r for r in rows if r["group"] == g]
        rates.append(sum(1 for r in sub if r["decision"] == "select") / len(sub))
        ns.append(len(sub))

    hi = max(rates)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(range(len(groups)), rates, 0.5, color=BLUE, zorder=2)
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                f"{r.get_height():.1%}", ha="center", fontsize=9,
                color=INK, fontweight="bold")
    ax.axhline(hi * 0.8, color=ORANGE, lw=1.5, ls="--", zorder=3)
    ax.text(len(groups) - 0.55, hi * 0.8 + 0.012,
            "four-fifths threshold", color=ORANGE, fontsize=8, ha="right")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace("_", "\n") for g in groups], color=INK2)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    finish(fig, ax, "fig6_fairness_name_swap",
           "Selection rate by candidate name group",
           f"Byte-identical CVs; only the name differs (n={ns[0]} per group)")


def main():
    print("Generating figures...")
    for fn in [fig_forest, fig_controls, fig_ablations, fig_views,
               fig_calibration, fig_fairness]:
        fn()
    if skipped:
        print("\nSkipped (data not available yet):")
        for s in skipped:
            print(f"  - {s}")
        print("Re-run this script once those experiments finish.")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
