# Meeting brief — Sadia

*2026-08-21. Numbers: `RESULTS.md`. Full reasoning: `EVALUATION_AND_APPROACH_PLAN.md`. Figures: `figures/`.*

---

## If you remember only three things

1. **The dataset is broken, and I can prove it.** The hiring decisions in it
   mostly don't depend on what's in the CV. So no system — mine, Amol's, or
   anyone's — can score well on it.
2. **The retrieval part of my system does nothing.** I tested this properly.
   Giving the model *random* past cases works just as well as giving it the
   *most similar* ones. Giving it none at all works slightly better.
3. **That's a real result, not a failure.** We asked "does RAG help here?" and
   the answer is no, with the evidence to back it up.

---

## How to open

> "The pipeline is finished and fully evaluated. The main finding is that on
> this dataset, all the retrieval machinery adds nothing — the model does just
> as well with no retrieval at all. I can show exactly why that happens. It
> means the project should be about how you *evaluate* these systems, not about
> which retrieval method wins."

Start with the finding, not the accuracy number. If you say "we got 55%" first,
people hear a bad grade. If you explain the finding first, the 55% becomes
proof of it.

---

## Finding 1 — the dataset doesn't support the task

**In one line:** the hiring decisions in this dataset are close to random with
respect to the candidate's CV.

Three pieces of evidence:

**The best possible score is about 58%, not 100%.**
I trained a standard machine-learning model on all 10,174 records, *giving it
the correct answers to learn from*. That's the strongest anything can do with
this data. It only reached 58.2%. Random guessing gets 50%. So there are only
about 8 points of real signal in the whole dataset.

**The stated reasons don't match the CVs.**
Each record has a reason, like "lacks hands-on experience with cloud platforms."
I checked whether those reasons describe the actual candidates. They don't. Of
the people rejected *specifically* for lacking cloud experience, 29.4% mention
cloud in their CV. Across everyone else in the dataset, 33.1% mention it. So
the people supposedly rejected for missing cloud skills mention cloud slightly
*less often than average* — which is to say, the reason has nothing to do with
them.

**The job descriptions are nearly empty.**
Average length is 53 words. Only 21.7% mention any actual requirement. A real
example: *"We're hiring an E-commerce Specialist to develop and deliver
high-quality solutions to transform our healthcare."* There is nothing in there
to match a CV against.

---

## Finding 2 — retrieval adds nothing, and here's why

**In one line:** the past cases we retrieve and show the model don't change its
decisions.

I ran the same 300 candidates four times. Same model, same prompt. The only
thing I changed was what past cases the model was shown.

| What the model was shown | Accuracy |
|---|---|
| The 10 most similar past cases (our real system) | 54.3% |
| 10 **random** past cases from the same job role | 55.0% |
| 10 **random** past cases from anywhere | 55.0% |
| **Nothing at all** | **55.7%** |

All four are the same within measurement error. Showing the model carefully
retrieved cases is no better than showing it random ones — or nothing.

**Why this happens** (and this is the part worth explaining):

The idea behind our system is "show the model similar past candidates and what
happened to them." That only helps if similar candidates tended to get the
*same outcome*. I measured that directly, with no LLM involved.

When we retrieve the 10 most similar past cases, only **56.6%** of them share
the outcome of the person we're deciding about. If retrieval were useless it
would be 50%. So there are only **6.6 points of useful signal** in the evidence
before the model even looks at it.

**What to say:** that's why better retrieval can't help. Reranking, hybrid
search, fine-tuned embeddings — they're all fighting over those 6.6 points.

---

## Finding 3 — the model does read the CV

**In one line:** our system isn't lazily approving everyone; it responds
strongly to the CV, but that doesn't translate into being right.

I ran the same 300 candidates again, damaging the CV each time:

| What the model was given | It said "select" |
|---|---|
| The real CV | **87%** of the time |
| A *different* candidate's CV | 33% |
| No CV at all | **5.7%** |

The more real candidate information it gets, the more it approves. Remove the
CV and it rejects almost everyone. So it is genuinely reading and reacting to
the CV.

**What to say:** the model reacts strongly to the CV and *still* only scores
55.7%. That's the clearest evidence the problem is the dataset's labels, not our
system.

---

## What I built

- **Steps 5, 6 and 7** — the LLM decision engine, the evaluation harness, and
  the candidate rejection-feedback generator (a required deliverable that had no
  code at all before).
- **The comparison baselines the proposal requires**, including the traditional
  keyword-matching ATS. It scores 47.7% — **no better than a coin flip**, because
  there's nothing in a 53-word job ad to keyword-match against.
- **The experiments above**, which isolate what the system actually responds to.
- **A fairness test** — same CV, swap only the candidate's name, see if the
  decision changes. Running now.
- **The human labelling tool** — 150 cases, one web page per person, works
  offline, saves as you go.
- **A "does it work at all" benchmark** — 500 made-up cases where the right
  answer is guaranteed knowable, to check our pipeline isn't simply broken.
- **Automatic report and chart generation**, so no number is ever copied by hand.

---

## Two mistakes I found in my own work — say these yourself

Better to volunteer them. Both were caught by measurement, which is itself an
argument for doing the evaluation properly.

**1. My index was cheating.** Step 3 was including the "reason for decision"
field in the data it searched over. That field basically states the answer. A
simple model can guess the outcome from that field alone 91.9% of the time.
I found it, removed it, rebuilt everything, and re-ran all the numbers.

**2. My results weren't reproducible.** The model was running with randomness
turned on by default. The exact same test, run twice, gave 45% and then 50%.
I fixed it so runs are now identical every time. It also means I don't trust
any comparison from the small early tests — that 5-point wobble was bigger than
most of the differences I was trying to measure.

**And one design decision I reversed.** The "two-view embedding" was my own
idea and my main technical contribution. I finally measured whether it helps.
It doesn't — it performs the same as the simpler version while being twice the
size. I removed it. The index went from 62 MB to 31 MB with no loss.

---

## The uncomfortable one — raise it before they do

The feedback generator wrote a rejection letter telling a DevOps candidate they
hadn't shown experience with containerisation, cloud platforms, or scripting.

**Their CV lists Kubernetes, AWS, Azure, GCP and Python.**

Three false statements about their own application, and none of them were even
requirements in the job ad. This is exactly the harm the project set out to
study, produced by our own system on the second try.

I've added automatic checks that catch it, and they distinguish two things:
a gap that's simply *unsupported*, versus one the CV actively *disproves*. Every
generated message is now checked before it could go anywhere.

**What to say:** if you ask a model to justify a decision whose reasons don't
match the candidate's data, it will invent reasons that sound plausible. That's
predictable, not surprising. Any real system doing this needs automatic checking.

---

## Still running (results in a few hours)

- Fairness — does swapping the candidate's name change the decision?
- Counterfactual — if I add a skill the job requires, does the decision improve?
- The sanity benchmark — does the pipeline work when the task is fair?
- One last CV test (shortened CV)

---

## What I need from this meeting

1. **We both need to label the gold set.** 150 candidates each, 2–3 hours, done
   independently without seeing each other's answers or the dataset's. Right now
   every automated comparison sits in a narrow band, so comparing against *human*
   judgement is the only measurement left that can tell a good system from a
   lucky one. This is the critical path.
2. **Agreement to reframe the project** — from "which retrieval method wins" to
   "how do you evaluate a hiring AI when the ground truth can't be trusted, and
   is it fair?" We need the supervisor to agree before we write up.
3. **Redirect Amol's retrieval work.** Better retrieval can't pay off here and I
   have the measurement that shows it. His time is better spent on the gold set
   and the fairness side.
4. **Decide whether to find a second dataset** with real hiring outcomes, to see
   whether these findings hold elsewhere.

---

## Questions you'll get, and what to say

**"So the project failed?"**
No. We asked whether RAG helps here, and the answer is no — that's a result. We
also measured *why*, which is the part that would apply to other datasets. A
leaderboard where every method scores 55% would have taught us nothing.

**"55% sounds bad."**
The best achievable on this data is about 58%, and simply saying "yes" to
everyone scores 50%. Everything anyone has built sits between 47.7% and 55.7%.

**"Why not use a bigger model?"**
Worth trying, and I've listed it as a limitation. But it wouldn't change
findings 1 or 2 — the ceiling and the missing signal are properties of the data,
measured without any LLM at all.

**"How sure are you that retrieval doesn't help?"**
It's 300 candidates, so the margin is about ±6 points. I'm careful to say "no
difference we can detect", not "proven identical". Two extra checks support it:
the four conditions shared no past cases at all, and the model's written
reasoning was different nearly every time. It's reading what we give it — it
just doesn't change its mind.

**"What's left to do?"**
The human labelling (needs both of us), the fairness numbers (running), and the
write-up. The results chapter is already outlined with the tables and charts in
place.

---

## Jargon, if it comes up

- **Accuracy** — how often the system's select/reject matches the dataset's.
- **Select rate** — how often it says "select". Ours says select ~85% of the
  time, which is why its accuracy is barely above chance on a 50/50 dataset.
- **Baseline** — a deliberately simple comparison. "Always say select" scores
  50% here, so anything below that is doing worse than nothing.
- **p-value** — the chance of seeing a difference this big if there were really
  no difference. Small (<0.05) means "probably real". Our retrieval comparisons
  came out at 0.50 and above, meaning "no difference we can detect".
- **Confidence interval** — the range the true value probably sits in. Ours are
  about ±6 points at 300 cases, which is why I don't rank things that are 1–2
  points apart.
- **Zero-shot** — asking the model with no examples attached.
- **Ablation** — deliberately removing part of the input to see what it was
  contributing.
