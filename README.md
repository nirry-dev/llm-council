# llm-council

**Multi-model consensus for local models.**

Ask a question to a *council* of models and get back one answer that has been
argued over, not just generated. Three stages:

1. **Propose** — each model answers independently, in parallel.
2. **Review** — each model critiques the *others'* answers, shown anonymously.
3. **Synthesize** — a chair model writes the final answer from the proposals and
   the reviews together.

Works with anything that speaks the OpenAI `/chat/completions` API — llama.cpp's
`server`, Ollama, LM Studio, vLLM, or the OpenAI API itself. Pure Python
standard library; no SDKs to install.

```python
from council import Member, run_council

members = [
    Member("qwen",  "http://localhost:8080/v1", "qwen2.5-7b-instruct"),
    Member("llama", "http://localhost:8081/v1", "llama-3.1-8b-instruct"),
]
result = run_council("Explain CRDTs to a backend engineer.", members, chair=members[0])
print(result.final)
```

---

## Why this exists

A single 7–14B model running locally is capable but streaky: it will nail one
question and quietly make something up on the next. Running the *same* prompt
through a few different models and reconciling the answers catches a lot of those
misses — one model's hallucination is rarely shared by the others, and made to
critique each other they surface it.

The interesting part isn't "ask three models and average." It's the **structure**
that makes the reconciliation honest, and that structure is the whole point of
this repo.

## How it works

**Blind peer review is the core idea.** When a model is told "this answer came
from a bigger/better model," it tends to defer to it — authority bias, same as
people. So before anyone reviews anything:

- proposals are stripped of their authors and relabeled `Candidate A`, `B`, `C`…
- the labels are **shuffled**, so position carries no signal either
- a model never reviews its own proposal

Each reviewer then has to judge answers on their merits and rank them. The chair
model gets the anonymized proposals *and* the reviews, and writes the final
answer — resolving disagreements, folding in corrections, and dropping whatever
review showed to be wrong.

```
 question ─┬─▶ qwen   ─┐                       ┌─▶ qwen reviews  B, C
           ├─▶ llama  ─┼─▶ shuffle + anonymize ┼─▶ llama reviews A, C ─┐
           └─▶ mistral┘   (A, B, C)            └─▶ mistral reviews A, B│
                                                                       ▼
                                             chair ◀── proposals + reviews ──▶ final answer
```

Everything runs in parallel where it can (all proposals at once, all reviews at
once), so wall-clock time is roughly two model round-trips plus the chair, not
`2N + 1` of them.

## Install

```bash
pip install llm-council        # once published
# or, from a checkout:
pip install -e .
```

No dependencies.

## Configure

A council is a list of members and an optional chair. See
[`council.example.json`](council.example.json):

```json
{
  "temperature": 0.7,
  "members": [
    { "name": "qwen",    "endpoint": "http://localhost:8080/v1", "model": "qwen2.5-7b-instruct" },
    { "name": "llama",   "endpoint": "http://localhost:8081/v1", "model": "llama-3.1-8b-instruct" },
    { "name": "mistral", "endpoint": "http://localhost:8082/v1", "model": "mistral-7b-instruct" }
  ],
  "chair": { "name": "chair", "endpoint": "http://localhost:8080/v1", "model": "qwen2.5-14b-instruct" }
}
```

A member may carry an optional `system` persona and an `api_key` (otherwise
`OPENAI_API_KEY` is used, and omitted entirely for local servers that need none).
Point three members at one endpoint with different models, or at three different
servers — both work.

## Use

**From the shell:**

```bash
council ask "What's the difference between a mutex and a semaphore?" -c council.json
council ask -c council.json -f question.txt --verbose    # show the full transcript
```

**From Python:**

```python
from council import Member, run_council

result = run_council(question, members, chair=members[0])
print(result.final)                 # just the answer
print(result.format(verbose=True))  # proposals + reviews + answer

for p in result.proposals:          # inspect the transcript
    print(p.label, p.author, p.text[:80])
```

## Design notes & limitations

- **Not a fact oracle.** The council reduces one-model flukes; it does not make a
  set of small models reliably correct. If every member shares a misconception,
  so will the result.
- **Cost is roughly 2N + 1 calls.** Three members is about seven completions.
  Parallelism keeps the *latency* near three sequential calls, but the token bill
  scales with the council size.
- **One review round.** Reviewers see proposals but not each other's reviews, and
  there's no second pass. A debate-style loop (propose → review → revise → review)
  is the obvious extension; one round was the point where quality jumped most for
  the least complexity in my testing.
- **The chair is a single point of taste.** The final answer is only as good as
  the chair's judgment. Using your strongest model as chair matters more than
  which models propose.

## Learning notes

*What I set out to learn, and what actually moved the needle.*

- **Structure beats raw model size, up to a point.** Three 7B models with blind
  review and a 14B chair gave noticeably steadier answers than the 14B alone —
  not because voting is magic, but because being forced to *critique* surfaces
  errors a model won't catch while generating. The protocol was doing more work
  than any single upgrade would have.
- **Anonymity is load-bearing, and I didn't expect that.** My first version
  labeled proposals with model names. Reviews collapsed into "GPT-style answer
  looks most authoritative." Stripping names and shuffling the order changed the
  reviews from deference to actual judgment — the single highest-leverage line in
  the codebase is the shuffle.
- **Make the pipeline inspectable or you can't tune it.** Returning the full
  transcript (every proposal, every review) wasn't a feature request — it was how
  I could *see* that a stage was underperforming and fix the prompt, instead of
  staring at a final answer and guessing.
- **Parallelism is free here, so take it.** The stages are embarrassingly
  parallel within themselves; a `ThreadPoolExecutor` turned `2N+1` sequential
  round-trips into about three, which is the difference between "fun to use" and
  "too slow to bother."
- **Dependency-free pays off at the edges.** Talking to the OpenAI API shape over
  `urllib` meant the same code drives llama.cpp, Ollama, and LM Studio with only
  a URL change — and the test suite needs no network at all, just a fake chat
  function.

## Tests

```bash
python -m unittest discover -s tests -v
```

A recorder stands in for the model, so the suite runs offline and asserts the
exact prompts each stage sends — including that reviewers never see their own
answer.

## License

MIT © Nirry
