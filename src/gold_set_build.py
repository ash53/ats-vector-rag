"""
GOLD SET — build blind labelling sheets for human annotators

Why this exists: every automated reference line on this dataset lands in a
50-55% band (always-select 50.0%, keyword TF-IDF 47.7% at AUC 0.474, supervised
ceiling 53.7%). Agreement with the dataset label therefore cannot tell us
whether a system is any good. Human labels can — and comparing human agreement
with the dataset measures how trustworthy the dataset is in the first place.

Design decisions:

* The gold cases are drawn from the SAME seed-42 sample the RAG pipeline is
  evaluated on, so every human-labelled case already has a system prediction.
* Annotators never see the decision or the reason. The sheet contains only what
  a real screener would have: role, job description, CV.
* Each annotator gets a different presentation order, so fatigue and drift do
  not correlate between them.
* REPEATS: a handful of cases appear twice for each annotator, at the start and
  again at the end. Disagreement with themselves is the annotator's own noise
  floor — the number that says how much of any human/system gap is just human
  inconsistency. Annotators are not told which cases repeat.

Output:
    data/gold/gold_cases.json          full truth-bearing record (do NOT open
                                       before labelling — it contains the labels)
    data/gold/label_sheet_sadia.html   blind, self-contained, works offline
    data/gold/label_sheet_amol.html

Usage:
    python src/gold_set_build.py                    # 150 cases + 10 repeats
    python src/gold_set_build.py --n-gold 60
"""

import json
import random
import argparse
import html
from pathlib import Path

from step6_evaluate import load_cases, stratified_sample

CASES_PATH = Path("data/processed/cases_stage2.jsonl")
OUT_DIR = Path("data/gold")

# The evaluation sample the gold set is carved out of. Must match the RAG run.
EVAL_N = 300
EVAL_SEED = 42

ANNOTATORS = {"sadia": 101, "amol": 202}   # name -> shuffle seed


def build_gold_cases(n_gold, n_repeats):
    all_cases = load_cases(CASES_PATH)
    eval_sample = stratified_sample(all_cases, EVAL_N, EVAL_SEED)

    # Balanced subset of the eval sample, taken deterministically in sample order
    sel = [c for c in eval_sample if c["decision"].lower() == "select"]
    rej = [c for c in eval_sample if c["decision"].lower() == "reject"]
    half = n_gold // 2
    gold = sel[:half] + rej[:half]

    rng = random.Random(7)
    repeats = rng.sample([c["case_id"] for c in gold], min(n_repeats, len(gold)))
    return gold, repeats


def build_items(gold, repeats, seed):
    """One annotator's presentation order. Repeats are shown early and again late."""
    rng = random.Random(seed)
    by_id = {c["case_id"]: c for c in gold}

    main = [c["case_id"] for c in gold]
    rng.shuffle(main)

    # Ensure the repeated cases appear in the first half, then again at the end
    head = [cid for cid in main if cid in repeats]
    rest = [cid for cid in main if cid not in repeats]
    rng.shuffle(rest)
    order = head + rest + head[:]        # trailing copies are the repeat probes

    items = []
    for pos, cid in enumerate(order):
        c = by_id[cid]
        items.append({
            "item_id": f"{pos:03d}",
            "case_id": cid,
            "role": c["role"],
            "jd_text": c["jd_text"],
            "cv_text": c["cv_text"],
        })
    return items


HTML_TEMPLATE = """<meta charset="utf-8">
<title>Gold set labelling — __NAME__</title>
<style>
  :root {
    --bg:#faf9f7; --fg:#1c1b19; --muted:#6b6862; --line:#e0ddd6;
    --card:#fff; --accent:#2f5d50; --sel:#2f5d50; --rej:#9b3b2f;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#171614; --fg:#eceae5; --muted:#9a968e; --line:#2f2d29;
            --card:#1f1e1b; --accent:#7fb3a3; --sel:#7fb3a3; --rej:#d98b7d; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
           padding:12px 24px; display:flex; gap:16px; align-items:center; z-index:10; }
  .bar { flex:1; height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
  .bar > div { height:100%; background:var(--accent); width:0; transition:width .2s; }
  main { max-width:820px; margin:0 auto; padding:24px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:20px 24px; margin-bottom:16px; }
  h2 { font-size:13px; letter-spacing:.08em; text-transform:uppercase;
       color:var(--muted); margin:0 0 10px; font-weight:600; }
  pre { white-space:pre-wrap; word-wrap:break-word; font:14px/1.65 ui-monospace,Menlo,monospace;
        margin:0; max-height:420px; overflow-y:auto; }
  .role { font-size:22px; font-weight:600; margin:0 0 4px; }
  .btns { display:flex; gap:12px; margin:20px 0 10px; }
  button { flex:1; padding:14px; font-size:16px; font-weight:600; cursor:pointer;
           border-radius:8px; border:1px solid var(--line); background:var(--card);
           color:var(--fg); }
  button.sel:hover, button.sel.on { background:var(--sel); color:var(--bg); border-color:var(--sel); }
  button.rej:hover, button.rej.on { background:var(--rej); color:var(--bg); border-color:var(--rej); }
  .conf { display:flex; gap:8px; margin-bottom:12px; }
  .conf button { padding:8px; font-size:14px; font-weight:500; }
  .conf button.on { background:var(--accent); color:var(--bg); border-color:var(--accent); }
  textarea { width:100%; min-height:60px; padding:10px; border-radius:8px;
             border:1px solid var(--line); background:var(--bg); color:var(--fg);
             font:14px/1.5 inherit; resize:vertical; }
  .nav { display:flex; justify-content:space-between; gap:12px; margin-top:16px; }
  .nav button { flex:0 0 auto; padding:10px 20px; font-weight:500; }
  .muted { color:var(--muted); font-size:14px; }
  .done { text-align:center; padding:60px 20px; }
  kbd { border:1px solid var(--line); border-radius:4px; padding:1px 6px;
        font:12px ui-monospace,monospace; }
</style>

<header>
  <strong>__NAME__</strong>
  <div class="bar"><div id="prog"></div></div>
  <span class="muted" id="count"></span>
  <button id="export" style="flex:0 0 auto;padding:8px 16px;font-size:14px">Export</button>
</header>

<main id="app"></main>

<script>
const ITEMS = __ITEMS__;
const NAME = "__NAME__";
const KEY = "goldset_" + NAME;
let answers = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;
const firstUnanswered = ITEMS.findIndex(t => !answers[t.item_id]);
if (firstUnanswered > 0) i = firstUnanswered;

function save() { localStorage.setItem(KEY, JSON.stringify(answers)); }

function render() {
  const n = ITEMS.length;
  const answered = Object.keys(answers).length;
  document.getElementById("prog").style.width = (100 * answered / n) + "%";
  document.getElementById("count").textContent = answered + " / " + n;

  if (i >= n) {
    document.getElementById("app").innerHTML =
      '<div class="done"><h1>Done — ' + answered + ' / ' + n + '</h1>' +
      '<p class="muted">Click Export and send the file to the other annotator.</p></div>';
    return;
  }

  const t = ITEMS[i];
  const a = answers[t.item_id] || {};
  document.getElementById("app").innerHTML = `
    <p class="muted">Item ${i + 1} of ${n}</p>
    <p class="role">${t.role}</p>
    <div class="card"><h2>Job description</h2><pre>${t.jd_text}</pre></div>
    <div class="card"><h2>Candidate CV</h2><pre>${t.cv_text}</pre></div>
    <div class="btns">
      <button class="sel ${a.decision === "select" ? "on" : ""}" data-d="select">SELECT <kbd>s</kbd></button>
      <button class="rej ${a.decision === "reject" ? "on" : ""}" data-d="reject">REJECT <kbd>r</kbd></button>
    </div>
    <p class="muted" style="margin:4px 0">Confidence</p>
    <div class="conf">
      ${["high", "medium", "low"].map(c =>
        `<button data-c="${c}" class="${a.confidence === c ? "on" : ""}">${c}</button>`).join("")}
    </div>
    <textarea id="note" placeholder="Optional: what decided it for you?">${a.note || ""}</textarea>
    <div class="nav">
      <button id="prev" ${i === 0 ? "disabled" : ""}>&larr; Back</button>
      <span class="muted">Keys: <kbd>s</kbd> select · <kbd>r</kbd> reject · <kbd>&rarr;</kbd> next</span>
      <button id="next">Next &rarr;</button>
    </div>`;

  document.querySelectorAll("[data-d]").forEach(b => b.onclick = () => choose(b.dataset.d));
  document.querySelectorAll("[data-c]").forEach(b => b.onclick = () => {
    answers[t.item_id] = Object.assign({}, answers[t.item_id], {confidence: b.dataset.c});
    save(); render();
  });
  document.getElementById("note").onblur = e => {
    if (answers[t.item_id]) { answers[t.item_id].note = e.target.value; save(); }
  };
  document.getElementById("prev").onclick = () => { if (i > 0) { i--; render(); } };
  document.getElementById("next").onclick = () => { i++; render(); };
}

function choose(d) {
  const t = ITEMS[i];
  answers[t.item_id] = Object.assign({case_id: t.case_id, decision: d,
                                      ts: new Date().toISOString()},
                                     answers[t.item_id] || {}, {decision: d});
  save();
  setTimeout(() => { i++; render(); }, 120);
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "s") choose("select");
  if (e.key === "r") choose("reject");
  if (e.key === "ArrowRight") { i++; render(); }
  if (e.key === "ArrowLeft" && i > 0) { i--; render(); }
});

document.getElementById("export").onclick = () => {
  const out = {annotator: NAME, exported: new Date().toISOString(),
               n_items: ITEMS.length, answers: answers};
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "gold_labels_" + NAME + ".json";
  a.click();
};

render();
</script>
"""


def main(n_gold=150, n_repeats=10):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold, repeats = build_gold_cases(n_gold, n_repeats)
    print(f"gold cases: {len(gold)} "
          f"({sum(1 for c in gold if c['decision'].lower() == 'select')} select / "
          f"{sum(1 for c in gold if c['decision'].lower() == 'reject')} reject)")
    print(f"repeat probes per annotator: {len(repeats)}")

    truth = {
        "config": {"n_gold": len(gold), "n_repeats": len(repeats),
                   "eval_n": EVAL_N, "eval_seed": EVAL_SEED,
                   "annotators": list(ANNOTATORS)},
        "repeat_case_ids": repeats,
        "cases": [{"case_id": c["case_id"], "role": c["role"],
                   "decision": c["decision"].lower(),
                   "decision_reason": c["decision_reason"]} for c in gold],
    }
    (OUT_DIR / "gold_cases.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    for name, seed in ANNOTATORS.items():
        items = build_items(gold, repeats, seed)
        page = (HTML_TEMPLATE
                .replace("__ITEMS__", json.dumps(items))
                .replace("__NAME__", html.escape(name)))
        path = OUT_DIR / f"label_sheet_{name}.html"
        path.write_text(page, encoding="utf-8")
        print(f"  {path}  ({len(items)} items, {path.stat().st_size // 1024} KB)")

    print(f"\nTruth file: {OUT_DIR / 'gold_cases.json'} — do not open before labelling.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-gold", type=int, default=150)
    p.add_argument("--n-repeats", type=int, default=10)
    a = p.parse_args()
    main(n_gold=a.n_gold, n_repeats=a.n_repeats)
