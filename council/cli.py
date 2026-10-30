"""Command line: `council ask "question" --config council.json`.

    council ask "Explain CRDTs simply." -c council.json
    council ask -c council.json -f question.txt --verbose

The config lists the members and (optionally) a chair; see council.example.json.
"""
import argparse
import json
import sys

from .client import ChatError
from .council import Member, run_council

_MEMBER_FIELDS = {"name", "endpoint", "model", "system", "api_key"}


def _member(d):
    return Member(**{k: v for k, v in d.items() if k in _MEMBER_FIELDS})


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("members"):
        raise ValueError("config must list at least one member under 'members'")
    members = [_member(m) for m in cfg["members"]]
    chair = _member(cfg["chair"]) if cfg.get("chair") else members[0]
    return {
        "members": members,
        "chair": chair,
        "temperature": float(cfg.get("temperature", 0.7)),
        "max_tokens": int(cfg.get("max_tokens", 1024)),
    }


def cmd_ask(args):
    try:
        cfg = _load_config(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.question_file:
        with open(args.question_file, "r", encoding="utf-8") as f:
            question = f.read().strip()
    else:
        question = args.question
    if not question:
        print("no question given (pass it as an argument or with --question-file)", file=sys.stderr)
        return 2

    try:
        result = run_council(
            question, cfg["members"], cfg["chair"],
            temperature=cfg["temperature"], max_tokens=cfg["max_tokens"], seed=args.seed,
        )
    except ChatError as exc:
        print(f"council failed: {exc}", file=sys.stderr)
        return 1

    print(result.format(verbose=args.verbose))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="council",
        description="Multi-model consensus: propose, blind peer review, chair synthesis.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("ask", help="ask the council a question")
    pa.add_argument("question", nargs="?", help="the question (or use --question-file)")
    pa.add_argument("-c", "--config", required=True, help="council config JSON")
    pa.add_argument("-f", "--question-file", help="read the question from a file")
    pa.add_argument("-v", "--verbose", action="store_true",
                    help="also print every proposal and review")
    pa.add_argument("--seed", type=int, default=None, help="seed the anonymization shuffle")
    pa.set_defaults(func=cmd_ask)

    args = p.parse_args(argv)
    return args.func(args)
