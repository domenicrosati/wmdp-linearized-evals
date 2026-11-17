from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

from .protocol import metrics, sha256

RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
GD_LR_GRID = (0.03, 0.1, 0.3)
CHECKPOINTS = (0, 25, 50, 100, 200, 400)


def _predict(
    features: np.ndarray, base: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    return base + np.einsum(
        "nkd,d->nk", features, weight, optimize=True
    )


def _ridge_fit(
    features: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    alpha: float,
) -> np.ndarray:
    target = np.eye(4, dtype=np.float64)[labels] - base
    target -= target.mean(axis=1, keepdims=True)
    x = features[ids].reshape(-1, features.shape[-1]).astype(np.float64)
    y = target[ids].reshape(-1)
    return np.linalg.solve(
        x.T @ x + alpha * len(ids) * np.eye(x.shape[1]),
        x.T @ y,
    )


def _gd(
    features: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    construction: np.ndarray,
    validation: np.ndarray,
    report: np.ndarray,
    lr: float,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    x = torch.from_numpy(features)
    z0 = torch.from_numpy(base)
    y = torch.from_numpy(labels).long()
    ids = torch.from_numpy(construction)
    weight = torch.zeros(features.shape[-1], requires_grad=True)
    optimizer = torch.optim.SGD(
        [weight], lr=lr, weight_decay=1e-3
    )
    trajectory: list[dict] = []
    snapshots: dict[int, np.ndarray] = {}
    for step in range(max(CHECKPOINTS) + 1):
        if step in CHECKPOINTS:
            with torch.no_grad():
                logits = (
                    z0 + torch.einsum("nkd,d->nk", x, weight)
                ).cpu().numpy()
            trajectory.append(
                {
                    "step": step,
                    "construction": metrics(
                        logits, labels, construction
                    ),
                    "validation": metrics(logits, labels, validation),
                    "report": metrics(logits, labels, report),
                }
            )
            snapshots[step] = weight.detach().numpy().copy()
        if step == max(CHECKPOINTS):
            break
        optimizer.zero_grad(set_to_none=True)
        scores = z0[ids] + torch.einsum(
            "nkd,d->nk", x[ids], weight
        )
        torch.nn.functional.cross_entropy(scores, y[ids]).backward()
        optimizer.step()
    return trajectory, snapshots


def run_attacks(
    cache: Path, output: Path, arrays: Path, rank: int = 512
) -> dict:
    if output.exists() or arrays.exists():
        raise FileExistsError("refusing to overwrite result artifacts")
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    base = payload["base_scores"].float().numpy()
    hidden = payload["features"].float().numpy()
    labels = payload["labels"].numpy().astype(np.int64)
    split = {
        key: np.asarray(value, dtype=np.int64)
        for key, value in payload["split"].items()
    }
    construction = split["construction"]
    validation = split["validation"]
    report = split["report"]
    non_report = np.sort(
        np.concatenate([construction, validation])
    )

    centered = hidden - hidden.mean(axis=1, keepdims=True)
    fit = centered[non_report].reshape(-1, centered.shape[-1])
    actual_rank = min(rank, fit.shape[0] - 1, fit.shape[1])
    _, _, basis = randomized_svd(
        fit,
        n_components=actual_rank,
        n_iter=4,
        random_state=17,
    )
    features = np.einsum(
        "nkd,rd->nkr", centered, basis, optimize=True
    ).astype(np.float32)
    scale = np.maximum(
        np.sqrt(
            np.mean(features[construction] ** 2, axis=(0, 1))
        ),
        1e-6,
    ).astype(np.float32)
    features /= scale[None, None, :]

    initial = {
        key: metrics(base, labels, ids)
        for key, ids in (
            ("construction", construction),
            ("validation", validation),
            ("report", report),
        )
    }
    ridge_grid = []
    ridge_weights = {}
    for alpha in RIDGE_GRID:
        weight = _ridge_fit(
            features, base, labels, construction, alpha
        )
        logits = _predict(features, base, weight)
        row = {
            "alpha": alpha,
            "construction": metrics(
                logits, labels, construction
            ),
            "validation": metrics(logits, labels, validation),
            "report": metrics(logits, labels, report),
        }
        ridge_grid.append(row)
        ridge_weights[alpha] = weight
    selected_ridge = min(
        ridge_grid,
        key=lambda row: (
            row["validation"]["cross_entropy"],
            row["alpha"],
        ),
    )

    gd_grid = []
    gd_weights = {}
    for lr in GD_LR_GRID:
        trajectory, snapshots = _gd(
            features,
            base,
            labels,
            construction,
            validation,
            report,
            lr,
        )
        selected = min(
            trajectory,
            key=lambda row: (
                row["validation"]["cross_entropy"],
                row["step"],
            ),
        )
        gd_grid.append(
            {
                "learning_rate": lr,
                "trajectory": trajectory,
                "selected": selected,
            }
        )
        gd_weights[(lr, selected["step"])] = snapshots[
            selected["step"]
        ]
    selected_gd = min(
        gd_grid,
        key=lambda row: (
            row["selected"]["validation"]["cross_entropy"],
            row["learning_rate"],
            row["selected"]["step"],
        ),
    )

    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays,
        basis=basis.astype(np.float32),
        scale=scale,
        ridge_weight=ridge_weights[
            selected_ridge["alpha"]
        ].astype(np.float32),
        gd_weight=gd_weights[
            (
                selected_gd["learning_rate"],
                selected_gd["selected"]["step"],
            )
        ].astype(np.float32),
        construction_ids=construction,
        validation_ids=validation,
        report_ids=report,
    )
    result = {
        "experiment": "WMDP-LINEARIZED-REDTEAM-V1",
        "status": "completed",
        "scope": (
            "Frozen final answer-token features only. This does not "
            "certify resistance to LoRA, full fine-tuning, feature "
            "learning, richer sidecars, or other interfaces."
        ),
        "cache": str(cache),
        "cache_sha256": sha256(cache),
        "feature_rank": actual_rank,
        "ridge_grid": list(RIDGE_GRID),
        "gd_learning_rate_grid": list(GD_LR_GRID),
        "gd_checkpoints": list(CHECKPOINTS),
        "initial": initial,
        "ridge": {"selected": selected_ridge, "grid": ridge_grid},
        "linearized_ce_gd": {
            "selected": selected_gd,
            "grid": gd_grid,
        },
        "arrays": str(arrays),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    result["results_sha256"] = sha256(output)
    result["arrays_sha256"] = sha256(arrays)
    return result


def replay(cache: Path, results: Path, arrays: Path) -> dict:
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    result = json.loads(results.read_text())
    saved = np.load(arrays)
    base = payload["base_scores"].float().numpy()
    hidden = payload["features"].float().numpy()
    labels = payload["labels"].numpy().astype(np.int64)
    centered = hidden - hidden.mean(axis=1, keepdims=True)
    features = np.einsum(
        "nkd,rd->nkr", centered, saved["basis"], optimize=True
    ).astype(np.float32)
    features /= saved["scale"][None, None, :]
    report = saved["report_ids"].astype(np.int64)
    ridge_logits = _predict(
        features, base, saved["ridge_weight"]
    )
    gd_logits = _predict(features, base, saved["gd_weight"])
    ridge_metrics = metrics(ridge_logits, labels, report)
    gd_metrics = metrics(gd_logits, labels, report)
    expected_ridge = result["ridge"]["selected"]["report"]
    expected_gd = result["linearized_ce_gd"]["selected"]["report"]
    errors = {
        "ridge_accuracy": abs(
            ridge_metrics["accuracy"] - expected_ridge["accuracy"]
        ),
        "ridge_cross_entropy": abs(
            ridge_metrics["cross_entropy"]
            - expected_ridge["cross_entropy"]
        ),
        "gd_accuracy": abs(
            gd_metrics["accuracy"] - expected_gd["accuracy"]
        ),
        "gd_cross_entropy": abs(
            gd_metrics["cross_entropy"]
            - expected_gd["cross_entropy"]
        ),
    }
    return {
        "passed": max(errors.values()) <= 1e-8,
        "errors": errors,
        "cache_sha256_matches": (
            result["cache_sha256"] == sha256(cache)
        ),
        "ridge_report": ridge_metrics,
        "linearized_ce_gd_report": gd_metrics,
    }
