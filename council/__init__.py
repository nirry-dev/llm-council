"""llm-council — multi-model consensus for local models.

Ask a question to a *council* of models and get one answer back that has been
argued over. Three stages:

    1. propose    — each model answers independently, in parallel
    2. review     — each model critiques the others' answers, shown anonymously
    3. synthesize — a chair model writes the final answer from both

Model-agnostic: anything with an OpenAI-compatible `/chat/completions` endpoint
(llama.cpp server, Ollama, LM Studio, vLLM, or the OpenAI API). Standard library
only — no SDKs.

    from council import Member, run_council

    members = [
        Member("qwen",  "http://localhost:8080/v1", "qwen2.5-7b-instruct"),
        Member("llama", "http://localhost:8081/v1", "llama-3.1-8b-instruct"),
    ]
    result = run_council("Explain CRDTs simply.", members, chair=members[0])
    print(result.final)
"""
from .client import ChatError, chat
from .council import CouncilResult, Member, Proposal, Review, run_council

__version__ = "0.1.0"
__all__ = [
    "Member",
    "run_council",
    "CouncilResult",
    "Proposal",
    "Review",
    "chat",
    "ChatError",
]
