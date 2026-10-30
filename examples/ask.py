"""Run a council from Python.

Point the endpoints at your local servers (llama.cpp `server`, Ollama's
`/v1`, LM Studio, vLLM, ...), then:

    python examples/ask.py
"""
from council import Member, run_council

members = [
    Member("qwen", "http://localhost:8080/v1", "qwen2.5-7b-instruct"),
    Member("llama", "http://localhost:8081/v1", "llama-3.1-8b-instruct"),
    Member("mistral", "http://localhost:8082/v1", "mistral-7b-instruct"),
]

result = run_council(
    "A user reports our API is 'sometimes slow'. How would you investigate?",
    members,
    chair=members[0],
)

# The final answer, plus the transcript it came from.
print(result.format(verbose=True))
