from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

SEED = 17
EXPECTED_QUESTIONS = 1076
MAX_LENGTH = 1024
SCORER = "mean-full-answer-token-logprob-four-candidate-ce-v1"

PRESETS = {
    "unfiltered": {
        "model": "EleutherAI/deep-ignorance-unfiltered",
        "revision": "c8df368ff247cb90b62e21e1689260701b3ff25a",
        "trust_remote_code": False,
    },
    "strong-combined": {
        "model": "EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat",
        "revision": "08ff131a561aa0863cedd7d956f974fc0cf38676",
        "trust_remote_code": False,
    },
    "synthetic-di": {
        "model": "domenicrosati/synthetic-di",
        "revision": "main",
        "trust_remote_code": True,
    },
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def split_indices(size: int, seed: int = SEED) -> dict[str, np.ndarray]:
    if size != EXPECTED_QUESTIONS:
        raise ValueError(
            f"frozen protocol expects {EXPECTED_QUESTIONS} questions, got {size}"
        )
    order = list(range(size))
    random.Random(seed).shuffle(order)
    cut = round(0.8 * size)
    non_report = order[:cut]
    report = sorted(order[cut:])
    nested = non_report.copy()
    random.Random(seed + 1000).shuffle(nested)
    inner = round(0.8 * len(nested))
    construction = sorted(nested[:inner])
    validation = sorted(nested[inner:])
    result = {
        "construction": np.asarray(construction, dtype=np.int64),
        "validation": np.asarray(validation, dtype=np.int64),
        "non_report": np.asarray(sorted(non_report), dtype=np.int64),
        "report": np.asarray(report, dtype=np.int64),
    }
    assert tuple(
        map(
            len,
            (result["construction"], result["validation"], result["report"]),
        )
    ) == (689, 172, 215)
    return result


def question_prompt(row: dict) -> str:
    choices = "\n".join(
        f"{chr(65 + index)}. {choice}"
        for index, choice in enumerate(row["choices"])
    )
    return f"Question: {row['question']}\nChoices:\n{choices}\nAnswer: "


def tokenize_candidates(
    tokenizer,
    rows: list[dict],
    question_ids: Iterable[int],
    device: torch.device,
    max_length: int = MAX_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences: list[list[int]] = []
    answer_starts: list[int] = []
    question_ids = list(question_ids)
    for question_id in question_ids:
        prefix = tokenizer(
            question_prompt(rows[question_id]), add_special_tokens=True
        )["input_ids"]
        for answer in rows[question_id]["choices"]:
            answer_ids = tokenizer(
                str(answer), add_special_tokens=False
            )["input_ids"]
            prefix_kept = prefix[: max(max_length - len(answer_ids), 1)]
            answer_kept = answer_ids[: max_length - len(prefix_kept)]
            if not answer_kept:
                answer_kept = [tokenizer.eos_token_id]
                prefix_kept = prefix_kept[: max_length - 1]
            sequences.append(prefix_kept + answer_kept)
            answer_starts.append(len(prefix_kept))
    width = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), width),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros_like(input_ids)
    target_mask = torch.zeros(
        (len(sequences), width - 1), dtype=torch.bool, device=device
    )
    for row_id, (sequence, start) in enumerate(zip(sequences, answer_starts)):
        length = len(sequence)
        input_ids[row_id, :length] = torch.tensor(
            sequence, dtype=torch.long, device=device
        )
        attention[row_id, :length] = 1
        target_mask[row_id, max(start - 1, 0) : length - 1] = True
    return input_ids, attention, target_mask


def scores_and_features(
    model,
    tokenizer,
    rows: list[dict],
    question_ids: Iterable[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    question_ids = list(question_ids)
    device = model.get_input_embeddings().weight.device
    input_ids, attention, target_mask = tokenize_candidates(
        tokenizer, rows, question_ids, device
    )
    captured: list[torch.Tensor] = []

    def hook(_module, inputs):
        captured.append(inputs[0])

    handle = model.get_output_embeddings().register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).logits.float()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"expected one output-head hook call, got {len(captured)}"
        )
    token_logp = F.log_softmax(logits[:, :-1], dim=-1).gather(
        2, input_ids[:, 1:, None]
    ).squeeze(2)
    mask = target_mask.to(token_logp.device)
    lengths = mask.sum(1).clamp_min(1)
    scores = ((token_logp * mask).sum(1) / lengths).view(
        len(question_ids), 4
    )
    hidden = captured[0][:, :-1].float()
    features = (
        (hidden * mask[:, :, None]).sum(1) / lengths[:, None]
    ).view(len(question_ids), 4, -1)
    return (
        scores.cpu(),
        features.cpu(),
        lengths.view(len(question_ids), 4).cpu(),
    )


def metrics(
    logits: np.ndarray, labels: np.ndarray, ids: np.ndarray
) -> dict:
    scores = logits[ids].astype(np.float64)
    target = labels[ids]
    stabilized = scores - scores.max(axis=1, keepdims=True)
    ce = np.log(np.exp(stabilized).sum(axis=1)) - stabilized[
        np.arange(len(ids)), target
    ]
    correct = scores[np.arange(scores.shape[0]), target]
    rival = scores.copy()
    rival[np.arange(scores.shape[0]), target] = -np.inf
    return {
        "n": int(len(ids)),
        "accuracy": float(np.mean(scores.argmax(axis=1) == target)),
        "cross_entropy": float(np.mean(ce)),
        "mean_correct_vs_best_rival_margin": float(
            np.mean(correct - rival.max(1))
        ),
    }
