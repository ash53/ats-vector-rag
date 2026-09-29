# ats-vector-rag — session context

University group project (Sadia + Amol). This repo is the vector-RAG track;
Amol's graph track is `ats-graph-rag`, **read-only from here — never commit or
push to it**. Cross-repo status: `progress_overview.md`.

---

## What the project found (current, as of 2026-09-29)

Two things are true, and both belong in the write-up. The second is the stronger
finding and it is **newer than most of the documents in this repo**.

**1. The dataset cannot support the task.** A supervised classifier trained on
all 10,174 rows *with the labels* reaches only 58.2%. The stated reason for each
decision is uncorrelated with the candidate (10.5% predictable vs a 10.4%
baseline). Job descriptions average 53 words; 21.7% state any requirement.

**2. The decision layer is also defective.** On a generated benchmark where the
JD lists its requirements explicitly, the CV plainly evidences them or not, and
there is zero label noise, the pipeline scores **51.0% with a 99% select rate** —
against 78% for bag-of-words and 100% for an oracle. It is right on every
"select" case and wrong on nearly every "reject" case.

**The model's effective decision rule is "is there a plausible CV here?", not
"does this candidate meet the requirements?"** Three experiments converge on it:

- Counterfactual: injecting a required skill the CV lacks changed the decision in
  10/100 cases; removing one it has, 3/100.
- Input ablations: select rate 87% with a real CV, 5.7% with none — but 88.7%
  with the CV truncated to 200 characters.
- Sanity benchmark: 99% select whether the candidate covers 0 or 5 of 5 skills.

Supporting results: retrieval contributes nothing (retrieved ≈ random ≈ none, all
paired p ≥ 0.50); only 6.6 points of signal exist in retrieved evidence; the
keyword-ATS baseline is at chance (AUC 0.474).

---

## ⚠️ Documents whose framing is now wrong

These still argue the ceiling belongs to the dataset and the pipeline is sound.
The sanity benchmark contradicts that. **Fix before quoting them:**

`docs/meeting_brief.md` · `EVALUATION_AND_APPROACH_PLAN.md` (§0.4, §1.6) ·
`progress_overview.md` · `docs/results_chapter_skeleton.md` (§5.13)

`RESULTS.md`, `figures/` and `docs/session_handover.md` are current.

---

## Where to read

| File | What it holds |
|---|---|
| `docs/session_handover.md` | **Start here.** Full state, all results, what's next |
| `RESULTS.md` | Every number — **generated, never hand-edit** |
| `EVALUATION_AND_APPROACH_PLAN.md` | Reasoning, method rationale, ranked next steps |
| `docs/results_chapter_skeleton.md` | How the results become a dissertation chapter |
| `README.md` | Setup and what each script answers |

---

## Running things

```bash
./venv/bin/python src/step6_evaluate.py --n 300 --seed 42 --out data/processed/eval_results_n300.json
./venv/bin/python src/make_results_table.py    # regenerates RESULTS.md
./venv/bin/python src/make_figures.py          # regenerates figures/
```

Needs Ollama running with `llama3.1:8b`. Roughly 10-12s per case, so a 300-case
run is about an hour — launch detached with `nohup`, never in the foreground.

Pipeline: step1 cases → step2 entities (vocab+regex) → step3 embeddings+FAISS →
step4 retrieval → step5 LLM decision → step6 evaluation → step7 feedback.
Experiments: `baselines.py`, `exp_prompt_variants.py` (controls `c_*`, ablations
`a_*`, prompts `v*`), `exp_retrieval_views.py`, `exp_counterfactual.py`,
`sanity_benchmark.py`, `gold_set_*.py`.

---

## Gotchas that have already cost time

- **Temperature is pinned** (`step5.TEMPERATURE = 0`, fixed seed). It was not
  before 2026-08-09: the same prompt on the same 40 cases scored 45.0% then
  50.0%. Distrust any number in a doc dated earlier.
- **Three FAISS indexes exist deliberately.** `faiss_index.bin` is live (768-d,
  case view only). `faiss_index_twoview.bin` reproduces pre-2026-08-21 results.
  `faiss_index_LEAKY.bin` is the pre-fix index that embedded `decision_reason`.
  None are tracked — too large.
- **`exp_retrieval_views.py` reads the two-view index on purpose** and refuses
  anything not 1536-d; splitting the live 768-d index would yield two
  meaningless halves.
- **`RESULTS.md` and `figures/` are generated.** Regenerate, don't edit.
- **`data/processed/eval_*.json` is tracked; everything else there is not.**
- **Never let the LLM aggregate.** Given an explicit rule and its own counts, it
  contradicted itself on 30% of cases; moving the arithmetic into code recovered
  5 points.
- Results files are checkpointed per condition — a long run that dies partway
  keeps what it finished.

---

## Next steps, in order

1. **Fix the four stale documents above.**
2. **Is the defect the model or the prompt?** Re-run `sanity_benchmark.py --eval`
   with the v3 requirement-checklist prompt, or a larger model. 90% → the prompt
   is fixable; still 51% → it's the model. ~1 hour, either answer is publishable.
3. **Label the gold set** — `data/gold/label_sheet_{sadia,amol}.html`, 150 cases,
   both annotators, blind. Unstarted, and it is the only measurement left that
   compares the system to human judgement.
4. **Write up**, using the chapter skeleton.

Not worth doing on this dataset: cross-encoder reranking, hybrid retrieval,
fine-tuned embeddings. The 6.6-point measurement shows there is nothing for
better retrieval to improve.

---

## Working conventions

- Commit only when asked. Pushes go to `main` on `ash53/ats-vector-rag`.
- Don't hand-quote numbers into prose — cite `RESULTS.md` or regenerate it.
- State negative results plainly; most of this project's value is in them.
- Report `n` and a confidence interval with any accuracy. n=300 gives ~±6 points,
  so differences under that are not resolvable.
