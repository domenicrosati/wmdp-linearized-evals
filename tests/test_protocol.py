import numpy as np

from wmdp_linearized.protocol import metrics, split_indices


def test_split_is_disjoint_and_frozen():
    split = split_indices(1076)
    assert len(split["construction"]) == 689
    assert len(split["validation"]) == 172
    assert len(split["report"]) == 215
    assert not set(split["construction"]) & set(split["validation"])
    assert not set(split["non_report"]) & set(split["report"])


def test_metrics_known_case():
    logits = np.array(
        [[4.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]
    )
    labels = np.array([0, 2])
    result = metrics(logits, labels, np.array([0, 1]))
    assert result["accuracy"] == 0.5
    assert result["mean_correct_vs_best_rival_margin"] == 0.5
    assert result["cross_entropy"] > 1.5

