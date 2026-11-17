from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attack import replay, run_attacks
from .extract import extract_cache
from .protocol import PRESETS


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--model")
    parser.add_argument("--model-label")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size-questions", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )


def resolve_model(args) -> dict:
    if args.preset:
        preset = PRESETS[args.preset]
        return {
            "model_name": args.model or preset["model"],
            "model_label": args.model_label or args.preset,
            "revision": args.revision or preset["revision"],
            "trust_remote_code": (
                args.trust_remote_code
                or preset["trust_remote_code"]
            ),
        }
    if not args.model:
        raise SystemExit("provide --preset or --model")
    return {
        "model_name": args.model,
        "model_label": args.model_label or Path(args.model).name,
        "revision": args.revision or "main",
        "trust_remote_code": args.trust_remote_code,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WMDP native inference and frozen-feature red-team attacks"
        )
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True
    )
    extract = subparsers.add_parser("extract")
    add_model_arguments(extract)
    extract.add_argument("--cache", type=Path, required=True)

    attack = subparsers.add_parser("attack")
    attack.add_argument("--cache", type=Path, required=True)
    attack.add_argument("--output", type=Path, required=True)
    attack.add_argument("--arrays", type=Path)
    attack.add_argument("--rank", type=int, default=512)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--cache", type=Path, required=True)
    audit.add_argument("--results", type=Path, required=True)
    audit.add_argument("--arrays", type=Path)

    all_parser = subparsers.add_parser("all")
    add_model_arguments(all_parser)
    all_parser.add_argument("--output-dir", type=Path, required=True)
    all_parser.add_argument("--rank", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        model = resolve_model(args)
        value = extract_cache(
            **model,
            output=args.cache,
            cache_dir=args.cache_dir,
            batch_size_questions=args.batch_size_questions,
            dtype=args.dtype,
        )
    elif args.command == "attack":
        arrays = args.arrays or args.output.with_suffix(".npz")
        value = run_attacks(
            args.cache, args.output, arrays, args.rank
        )
    elif args.command == "audit":
        arrays = args.arrays or args.results.with_suffix(".npz")
        value = replay(args.cache, args.results, arrays)
        if not value["passed"] or not value[
            "cache_sha256_matches"
        ]:
            print(json.dumps(value, indent=2, sort_keys=True))
            raise SystemExit(1)
    else:
        model = resolve_model(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cache = args.output_dir / "cache.pt"
        results = args.output_dir / "results.json"
        arrays = args.output_dir / "results.npz"
        extraction = extract_cache(
            **model,
            output=cache,
            cache_dir=args.cache_dir,
            batch_size_questions=args.batch_size_questions,
            dtype=args.dtype,
        )
        attacks = run_attacks(
            cache, results, arrays, args.rank
        )
        audit = replay(cache, results, arrays)
        value = {
            "extraction": extraction,
            "attacks": attacks,
            "audit": audit,
        }
        if not audit["passed"]:
            raise RuntimeError("independent replay failed")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

