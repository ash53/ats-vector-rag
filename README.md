# ATS Vector RAG

A retrieval-augmented pipeline that decides whether a candidate should be
selected for a role, plus the evaluation framework built to work out whether it
actually does anything.

Part of a two-repo group project; the graph-based counterpart lives in
`ats-graph-rag`. Cross-repo status is in `progress_overview.md`.

---

## What we found

Short version, with the numbers in [`RESULTS.md`](RESULTS.md) and the reasoning
in [`EVALUATION_AND_APPROACH_PLAN.md`](EVALUATION_AND_APPROACH_PLAN.md):

1. **The dataset's labels are largely unlearnable from the candidate data.** A
   fully supervised classifier trained on all 10,174 rows reaches 58.2%. The
   stated reason for each decision is unpredictable from the CV (10.5% against a
   10.4% majority baseline), and job descriptions average 53 words with only
   21.7% stating any requirement.
2. **Every method lands in a 47.7–55.7% band** — RAG, keyword TF-IDF, supervised
   logistic regression, and the LLM with no retrieval at all.
3. **Retrieval contributes nothing measurable.** Random exemplars match
   similarity-retrieved ones, and no exemplars at all does marginally better
   (paired McNemar p ≥ 0.50 across every pair of conditions).
4. **The two-view embedding did not earn its cost** and was removed: the case
   view alone is statistically indistinguishable at half the dimension.

The project's contribution is therefore the evaluation framework more than the
pipeline: how to tell whether an LLM hiring system works when the ground truth
cannot be trusted.

---

## Setup

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
```

The LLM runs locally through [Ollama](https://ollama.com):

```bash
ollama pull llama3.1:8b     # decision engine, see src/step5_llm_decision.py
```

Place the [Kaggle AI Recruitment Pipeline dataset](https://www.kaggle.com/datasets/yaswanthkumary/ai-recruitment-pipeline-dataset/data)
at `data/raw/dataset.csv`.

> The dataset, the FAISS index and the intermediate JSONL are gitignored by
> size. `data/processed/eval_*.json` **is** tracked — those are the experimental
> record. Steps 1–3 regenerate everything else in about 15 minutes.

---

## Pipeline

```
data/raw/dataset.csv
  step1  case construction        -> cases_stage1.jsonl
  step2  entity extraction        -> cases_stage2.jsonl      (vocab + regex, no LLM)
  step3  embeddings + FAISS       -> faiss_index.bin         (~11 min on MPS)
  step4  retrieval + evidence     -> stratified exemplars
  step5  LLM decision             -> select/reject + reasoning
  step6  evaluation               -> accuracy, F1, select rate, confusion matrix
  step7  rejection feedback       -> candidate-facing message + safety checks
```

```bash
./venv/bin/python src/step1_cases.py
./venv/bin/python src/step2_entities.py
./venv/bin/python src/step3_embeddings_faiss.py
./venv/bin/python src/step6_evaluate.py --n 300 --seed 42 --out data/processed/eval_results_n300.json
```

Runs are greedy and seeded (`step5.TEMPERATURE = 0`), so they reproduce. They
did not always: Ollama defaults to temperature 0.8, and the same prompt on the
same 40 cases once scored 45.0% and then 50.0%.

---

## Experiments

Each writes to `data/processed/eval_*.json`; `make_results_table.py` turns those
into `RESULTS.md`.

| Script | Question |
|---|---|
| `baselines.py` | What do always-select, keyword TF-IDF and a supervised classifier score? |
| `exp_prompt_variants.py --variants c_*` | Does retrieval contribute anything? (retrieved / random / none) |
| `exp_prompt_variants.py --variants a_*` | Does the model read the CV at all? (swapped / empty / truncated) |
| `exp_prompt_variants.py --variants v0,v1,v2,v3` | Does prompt design change the decision? |
| `exp_retrieval_views.py` | Is the two-view embedding worth it? (no LLM) |
| `exp_counterfactual.py --mode counterfactual` | Does the decision move when the evidence does? |
| `exp_counterfactual.py --mode fairness` | Does the decision change when only the name changes? |
| `sanity_benchmark.py --build --eval` | Does the pipeline work on a task whose answer is knowable? |
| `gold_set_build.py` / `gold_set_score.py` | How do the systems compare to human judgement? |

```bash
./venv/bin/python src/make_results_table.py        # regenerate RESULTS.md
```

---

## Human gold set

`data/gold/label_sheet_<name>.html` — open in a browser, no server needed. 150
cases drawn from the same seed-42 sample the pipeline is evaluated on, so every
labelled case already has a system prediction. Blind: role, JD and CV only.
Keyboard-driven, autosaves, Export writes a JSON file. Drop that into
`data/gold/` and run `gold_set_score.py` for inter-annotator agreement, kappa,
and every system scored against human consensus rather than the dataset label.

---

## Repository map

```
src/                      pipeline steps 1-7, experiments, gold-set tooling
data/raw/                 dataset.csv                              (gitignored)
data/processed/           cases, FAISS index                       (gitignored)
data/processed/eval_*.json  experimental record                    (tracked)
data/gold/                blind labelling sheets + truth file
data/vocab/               skills.txt, certifications.txt

RESULTS.md                every number, generated — do not hand-edit
EVALUATION_AND_APPROACH_PLAN.md   interpretation, caveats, what to do next
progress_overview.md      cross-repo status (this repo and ats-graph-rag)
MEETING_NOTES.md          point-in-time record, 2026-05-12
OVERVIEW.md / ROADMAP.md  superseded, see the two documents above
```

Two index files are kept deliberately:
`faiss_index_twoview.bin` (1536-d) reproduces everything committed before the
skill view was dropped, and `faiss_index_LEAKY.bin` is the pre-fix index that
embedded `decision_reason` — kept so the effect of that leak stays measurable.
