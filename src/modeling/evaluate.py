from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )

    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
    }