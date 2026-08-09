"""
GOLD SET — score human labels, then score the systems against the humans

Run after both annotators have exported their JSON from the labelling sheet
into data/gold/.

What it reports, in the order the argument needs to be made:

1. Each annotator's own noise floor, from the repeated cases. If a human
   disagrees with themselves on 15% of cases, no system can be expected to
   agree with them more than ~85% of the time.
2. Inter-annotator agreement (raw + Cohen's kappa). This is the human ceiling:
   the realistic best score any system could achieve on this task.
3. Each annotator against the DATASET label. If humans agree with each other
   far more than either agrees with the dataset, the dataset label is the
   unreliable party — which is the central claim of this project.
4. Every system we have results for, scored twice: against the dataset label,
   and against the human consensus (cases where both annotators agree).

Usage:
    python src/gold_set_score.py
"""

import json
from pathlib import Path

GOLD_DIR = Path("data/gold")
PROC_DIR = Path("data/processed")


def _kappa(a, b):
    """Cohen's kappa for two lists of binary labels."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in {"select", "reject"})
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _interpret_kappa(k):
    if k != k:
        return ""
    for lo, txt in [(0.8, "almost perfect"), (0.6, "substantial"), (0.4, "moderate"),
                    (0.2, "fair"), (0.0, "slight")]:
        if k >= lo:
            return f"({txt})"
    return "(worse than chance)"


def load_annotations():
    """Returns {annotator: {case_id: label}} plus self-consistency per annotator."""
    out, selfcons = {}, {}
    for path in sorted(GOLD_DIR.glob("gold_labels_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data["annotator"]
        seen, labels, repeats = {}, {}, []
        # answers are keyed by item_id, in presentation order
        for item_id in sorted(data["answers"]):
            ans = data["answers"][item_id]
            cid, dec = ans.get("case_id"), ans.get("decision")
            if not cid or not dec:
                continue
            if cid in seen:
                repeats.append((seen[cid], dec))      # (first pass, second pass)
            else:
                seen[cid] = dec
                labels[cid] = dec
        out[name] = labels
        selfcons[name] = repeats
        print(f"  {name}: {len(labels)} cases labelled, {len(repeats)} repeat probes "
              f"({path.name})")
    return out, selfcons


def main():
    truth_path = GOLD_DIR / "gold_cases.json"
    if not truth_path.exists():
        raise SystemExit("data/gold/gold_cases.json missing — run gold_set_build.py first")
    gold = json.loads(truth_path.read_text(encoding="utf-8"))
    dataset_label = {c["case_id"]: c["decision"] for c in gold["cases"]}

    print("Loading annotations...")
    ann, selfcons = load_annotations()
    if not ann:
        raise SystemExit(
            "\nNo gold_labels_*.json found in data/gold/.\n"
            "Open data/gold/label_sheet_<name>.html in a browser, label the cases, "
            "click Export, and save the download into data/gold/.")

    print("\n" + "=" * 72)
    print("1. ANNOTATOR SELF-CONSISTENCY (the human noise floor)")
    print("=" * 72)
    for name, pairs in selfcons.items():
        if not pairs:
            print(f"  {name}: no repeat probes completed")
            continue
        agree = sum(1 for a, b in pairs if a == b)
        print(f"  {name}: {agree}/{len(pairs)} = {agree/len(pairs):.0%} "
              f"agreement with own earlier judgement")

    names = sorted(ann)
    consensus = {}
    if len(names) >= 2:
        a, b = names[0], names[1]
        shared = sorted(set(ann[a]) & set(ann[b]))
        la = [ann[a][c] for c in shared]
        lb = [ann[b][c] for c in shared]
        raw = sum(1 for x, y in zip(la, lb) if x == y) / len(shared) if shared else 0
        k = _kappa(la, lb)

        print("\n" + "=" * 72)
        print("2. INTER-ANNOTATOR AGREEMENT (the human ceiling)")
        print("=" * 72)
        print(f"  {a} vs {b} on {len(shared)} shared cases")
        print(f"  raw agreement : {raw:.1%}")
        print(f"  Cohen's kappa : {k:.3f} {_interpret_kappa(k)}")
        print(f"\n  No system should be expected to beat {raw:.1%} — that is how "
              f"often\n  two humans looking at the same CV even agree with each other.")

        consensus = {c: ann[a][c] for c in shared if ann[a][c] == ann[b][c]}
        print(f"\n  consensus set: {len(consensus)} cases where both agree")

    print("\n" + "=" * 72)
    print("3. HUMANS vs THE DATASET LABEL")
    print("=" * 72)
    for name in names:
        shared = [c for c in ann[name] if c in dataset_label]
        agree = sum(1 for c in shared if ann[name][c] == dataset_label[c])
        sel = sum(1 for c in shared if ann[name][c] == "select") / len(shared)
        print(f"  {name:<8} agrees with dataset on {agree}/{len(shared)} = "
              f"{agree/len(shared):.1%}   (says select {sel:.0%} of the time)")
    if consensus:
        agree = sum(1 for c in consensus if consensus[c] == dataset_label[c])
        print(f"  {'both':<8} agree with dataset on {agree}/{len(consensus)} = "
              f"{agree/len(consensus):.1%}  (on the consensus set)")
        print("\n  If humans agree with each other far more than with the dataset,\n"
              "  the dataset label is the unreliable party.")

    # ------------------------------------------------------------------
    # 4. Systems
    # ------------------------------------------------------------------
    systems = {}

    for path, key in [(PROC_DIR / "eval_results_n300.json", None),
                      (PROC_DIR / "eval_results_n40.json", None)]:
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            systems[f"RAG ({path.stem.split('_')[-1]})"] = {
                r["case_id"]: r["prediction"] for r in d["results"]}

    p = PROC_DIR / "eval_prompt_variants.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, v in d["variants"].items():
            systems[f"prompt {k}"] = {r["case_id"]: r["prediction"] for r in v["results"]}

    for p in [PROC_DIR / "eval_baselines_n300.json", PROC_DIR / "eval_baselines_n40.json"]:
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            for k, v in d["baselines"].items():
                systems.setdefault(k, {r["case_id"]: r["prediction"] for r in v["results"]})

    if not systems:
        print("\n(no system result files found yet — run step6/baselines first)")
        return

    print("\n" + "=" * 72)
    print("4. SYSTEMS — scored against the dataset AND against human consensus")
    print("=" * 72)
    print(f"  {'system':<26} {'vs dataset':>12} {'vs humans':>12} {'n':>6}")
    print("  " + "-" * 58)
    for name, preds in systems.items():
        overlap = [c for c in preds if c in consensus] if consensus else []
        vs_data_ids = [c for c in preds if c in dataset_label]
        vs_data = (sum(1 for c in vs_data_ids if preds[c] == dataset_label[c])
                   / len(vs_data_ids)) if vs_data_ids else float("nan")
        if overlap:
            vs_hum = sum(1 for c in overlap if preds[c] == consensus[c]) / len(overlap)
            print(f"  {name:<26} {vs_data:>11.1%} {vs_hum:>12.1%} {len(overlap):>6}")
        else:
            print(f"  {name:<26} {vs_data:>11.1%} {'—':>12} {0:>6}")

    print("\n  'vs humans' is the number that means something. A system can only be\n"
          "  credited with a correct decision if two people independently agreed on it.")


if __name__ == "__main__":
    main()
