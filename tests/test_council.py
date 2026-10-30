"""Tests for the council protocol, using a fake chat function.

No network and no models required: a recorder stands in for the LLM so we can
assert on the exact prompts each stage sends and how the stages compose.
"""
import unittest

from council.council import Member, run_council


class Recorder:
    """A stand-in chat_fn that records every call and returns a tagged reply."""

    def __init__(self):
        self.calls = []

    def __call__(self, endpoint, model, messages, **kw):
        self.calls.append((model, messages))
        return f"<{model}> says hello"


def three_members():
    return [
        Member("alpha", "http://x/v1", "m1"),
        Member("beta", "http://x/v1", "m2"),
        Member("gamma", "http://x/v1", "m3"),
    ]


def is_review(messages):
    content = messages[-1]["content"]
    return "Candidate" in content and "Peer reviews" not in content


class CouncilTest(unittest.TestCase):
    def test_three_stage_flow(self):
        rec = Recorder()
        members = three_members()
        chair = Member("chair", "http://x/v1", "mc")
        result = run_council("Q?", members, chair, chat_fn=rec, seed=1)

        # 3 proposals + 3 reviews + 1 synthesis
        self.assertEqual(len(rec.calls), 7)
        self.assertEqual(len(result.proposals), 3)
        self.assertEqual(len(result.reviews), 3)
        self.assertEqual({p.author for p in result.proposals}, {"alpha", "beta", "gamma"})
        self.assertEqual(sorted(p.label for p in result.proposals), ["A", "B", "C"])
        self.assertEqual(result.final, "<mc> says hello")

    def test_reviews_are_blind_to_own_answer(self):
        rec = Recorder()
        members = three_members()
        run_council("Q?", members, chair=members[0], chat_fn=rec, seed=2)

        proposal_text = {m.model: f"<{m.model}> says hello" for m in members}
        review_calls = [(model, msgs) for model, msgs in rec.calls if is_review(msgs)]
        self.assertEqual(len(review_calls), 3)

        for model, msgs in review_calls:
            content = msgs[-1]["content"]
            # A reviewer never sees its own proposal...
            self.assertNotIn(proposal_text[model], content)
            # ...but does see both of the others.
            for other in (p for mdl, p in proposal_text.items() if mdl != model):
                self.assertIn(other, content)

    def test_single_member_skips_review(self):
        rec = Recorder()
        solo = [Member("solo", "http://x/v1", "m1")]
        result = run_council("Q?", solo, chair=solo[0], chat_fn=rec, seed=0)

        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.reviews, [])
        # Only a proposal and the chair synthesis — no review round.
        self.assertEqual(len(rec.calls), 2)

    def test_anonymization_is_seeded(self):
        # Same seed → same label/author mapping; a different seed may differ.
        a = run_council("Q?", three_members(), chair=three_members()[0],
                        chat_fn=Recorder(), seed=5)
        b = run_council("Q?", three_members(), chair=three_members()[0],
                        chat_fn=Recorder(), seed=5)
        self.assertEqual(
            [(p.label, p.author) for p in a.proposals],
            [(p.label, p.author) for p in b.proposals],
        )


if __name__ == "__main__":
    unittest.main()
