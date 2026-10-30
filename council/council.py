"""The council protocol: propose, blind peer review, chair synthesis.

Why blind review? When a model sees "this answer is from GPT-4" it tends to
defer to it; when it sees "Candidate B" it has to judge the content. So
proposals are stripped of author names and shuffled before anyone reviews them,
and a model never reviews its own answer.
"""
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional

from .client import chat as default_chat


@dataclass
class Member:
    """A council member (or the chair): a named model behind an endpoint."""
    name: str
    endpoint: str
    model: str
    system: Optional[str] = None     # optional persona for the propose stage
    api_key: Optional[str] = None


@dataclass
class Proposal:
    label: str       # anonymous label shown to reviewers, e.g. "A"
    author: str      # member name (kept for the transcript, hidden from models)
    text: str


@dataclass
class Review:
    reviewer: str
    text: str


@dataclass
class CouncilResult:
    question: str
    proposals: List[Proposal]
    reviews: List[Review]
    final: str

    def format(self, verbose=False):
        """Render the final answer, optionally with the full transcript."""
        if not verbose:
            return self.final
        out = ["# Proposals\n"]
        for p in self.proposals:
            out.append(f"## Candidate {p.label}  ({p.author})\n{p.text}\n")
        if self.reviews:
            out.append("# Reviews\n")
            for r in self.reviews:
                out.append(f"## {r.reviewer}\n{r.text}\n")
        out.append("# Final answer\n")
        out.append(self.final)
        return "\n".join(out)


_PROPOSE_SYS = (
    "You are an expert answering a question as one member of a council of experts. "
    "Give your best, complete answer. Be accurate and specific, and if you are "
    "genuinely uncertain about something, say so briefly rather than guessing."
)
_REVIEW_SYS = (
    "You are reviewing anonymous candidate answers from other experts. For each "
    "candidate, note real strengths and any errors, omissions, or dubious claims — "
    "judge correctness first, then completeness and clarity. Finish with a single "
    "ranking line from best to worst by label, e.g. 'Ranking: B > A > C'."
)
_CHAIR_SYS = (
    "You are the chair of a council of experts. Using the candidate answers and the "
    "peer reviews below, write the single best final answer to the question. Resolve "
    "disagreements on the merits, fold in corrections raised in review, and drop "
    "anything shown to be wrong. Do not mention the council, the candidates, the "
    "labels, or this process — just give the final answer directly."
)


def _propose_messages(question, member):
    return [
        {"role": "system", "content": member.system or _PROPOSE_SYS},
        {"role": "user", "content": question},
    ]


def _review_messages(question, others):
    blocks = "\n\n".join(f"--- Candidate {p.label} ---\n{p.text}" for p in others)
    return [
        {"role": "system", "content": _REVIEW_SYS},
        {"role": "user", "content": f"Question:\n{question}\n\nCandidate answers:\n\n{blocks}"},
    ]


def _synthesize_messages(question, proposals, reviews):
    pblocks = "\n\n".join(f"--- Candidate {p.label} ---\n{p.text}" for p in proposals)
    rblocks = "\n\n".join(f"--- Review {i + 1} ---\n{r.text}" for i, r in enumerate(reviews))
    user = f"Question:\n{question}\n\nCandidate answers:\n\n{pblocks}"
    if reviews:
        user += f"\n\nPeer reviews:\n\n{rblocks}"
    return [
        {"role": "system", "content": _CHAIR_SYS},
        {"role": "user", "content": user},
    ]


def run_council(question, members, chair, chat_fn=None, temperature=0.7,
                max_tokens=1024, seed=None):
    """Run the three-stage council and return a `CouncilResult`.

    `chat_fn(endpoint, model, messages, **kw)` is injectable for testing; it
    defaults to the real HTTP client.
    """
    chat_fn: Callable = chat_fn or default_chat
    members = list(members)
    if not members:
        raise ValueError("a council needs at least one member")

    # Stage 1 — independent proposals, in parallel.
    def propose(m):
        return chat_fn(m.endpoint, m.model, _propose_messages(question, m),
                       api_key=m.api_key, temperature=temperature, max_tokens=max_tokens)

    texts = _parallel(propose, members)

    # Anonymize: shuffle so neither label order nor author can bias reviewers.
    order = list(range(len(members)))
    random.Random(seed).shuffle(order)
    labels = [chr(ord("A") + i) for i in range(len(members))]
    proposals = [
        Proposal(label=label, author=members[idx].name, text=texts[idx])
        for label, idx in zip(labels, order)
    ]
    proposals.sort(key=lambda p: p.label)

    # Stage 2 — blind peer review: each member reviews everyone but themselves.
    def review(m):
        others = [p for p in proposals if p.author != m.name]
        if not others:
            return None
        return chat_fn(m.endpoint, m.model, _review_messages(question, others),
                       api_key=m.api_key, temperature=temperature, max_tokens=max_tokens)

    review_texts = _parallel(review, members)
    reviews = [Review(m.name, t) for m, t in zip(members, review_texts) if t is not None]

    # Stage 3 — the chair synthesizes the final answer.
    final = chat_fn(chair.endpoint, chair.model,
                    _synthesize_messages(question, proposals, reviews),
                    api_key=chair.api_key, temperature=temperature, max_tokens=max_tokens)

    return CouncilResult(question=question, proposals=proposals, reviews=reviews, final=final)


def _parallel(fn, items):
    """Map `fn` over `items` concurrently, preserving input order."""
    with ThreadPoolExecutor(max_workers=max(1, len(items))) as pool:
        return list(pool.map(fn, items))
