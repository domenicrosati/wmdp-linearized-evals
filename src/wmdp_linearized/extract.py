from __future__ import annotations

from pathlib import Path

import torch

from .protocol import (
    SCORER,
    canonical_json_sha256,
    scores_and_features,
    split_indices,
)


def load_wmdp_bio() -> tuple[list[dict], str]:
    from datasets import load_dataset

    dataset = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    rows = [
        {
            "question": row["question"],
            "choices": list(row["choices"]),
            "answer": int(row["answer"]),
        }
        for row in dataset
    ]
    return rows, canonical_json_sha256(rows)


def extract_cache(
    *,
    model_name: str,
    model_label: str,
    revision: str,
    output: Path,
    cache_dir: Path | None,
    trust_remote_code: bool,
    batch_size_questions: int,
    dtype: str,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows, dataset_sha = load_wmdp_bio()
    split = split_indices(len(rows))
    labels = torch.tensor(
        [row["answer"] for row in rows], dtype=torch.long
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    common = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
    }
    if cache_dir is not None:
        common["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        **common,
    ).cuda().eval()

    score_rows: list[torch.Tensor | None] = [None] * len(rows)
    feature_rows: list[torch.Tensor | None] = [None] * len(rows)
    length_rows: list[torch.Tensor | None] = [None] * len(rows)
    traversal = [
        *split["construction"].tolist(),
        *split["validation"].tolist(),
        *split["report"].tolist(),
    ]
    for start in range(0, len(traversal), batch_size_questions):
        ids = traversal[start : start + batch_size_questions]
        scores, features, lengths = scores_and_features(
            model, tokenizer, rows, ids
        )
        for offset, question_id in enumerate(ids):
            score_rows[question_id] = scores[offset]
            feature_rows[question_id] = features[offset]
            length_rows[question_id] = lengths[offset]
    if any(
        value is None
        for value in score_rows + feature_rows + length_rows
    ):
        raise RuntimeError("incomplete extraction")
    payload = {
        "base_scores": torch.stack(score_rows),  # type: ignore[arg-type]
        "features": torch.stack(feature_rows),  # type: ignore[arg-type]
        "answer_token_lengths": torch.stack(length_rows),  # type: ignore[arg-type]
        "labels": labels,
        "split": {key: value.tolist() for key, value in split.items()},
        "provenance": {
            "model": model_name,
            "model_label": model_label,
            "revision": revision,
            "dtype": dtype,
            "batch_size_questions": batch_size_questions,
            "scorer": SCORER,
            "dataset": "cais/wmdp:wmdp-bio:test",
            "dataset_canonical_sha256": dataset_sha,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    native_accuracy = float(
        (payload["base_scores"].argmax(1) == labels).float().mean()
    )
    return {
        "cache": str(output),
        "native_full_pool_accuracy": native_accuracy,
    }
