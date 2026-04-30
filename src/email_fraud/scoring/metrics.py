"""Shared verification metrics used by evaluation scripts.

All metrics follow the PAN Author Verification task conventions.
Labels are binary: 1 = authentic (email matches claimed sender), 0 = fraud.
Scores are continuous in [0, 1]: higher = more likely authentic.

PAN metrics
-----------
AUC   : area under the ROC curve — threshold-independent ranking quality.
EER   : equal error rate — the false-accept rate when FAR == FRR.
        Lower is better; a random classifier has EER = 0.5.
c@1   : PAN 2011 metric that rewards abstaining over guessing wrong.
        Unanswered questions (score == threshold) count partially toward the
        accuracy, based on the answered accuracy.  Encourages conservative models.
F0.5u : PAN 2019 metric with β=0.5, which weights precision over recall and
        also penalizes unanswered questions.
"""

from __future__ import annotations

import numpy as np


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC: probability that a genuine email scores higher than a fraud."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate: threshold where FAR == FRR.

    FAR (false accept rate) = FPR = false positives / (false positives + true negatives)
    FRR (false reject rate) = FNR = false negatives / (false negatives + true positives)
    EER is the operating point where the two are equal — a common single-number
    summary for biometric / authentication systems.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    # Find the index where FNR and FPR are closest; average the two for the EER estimate.
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_c_at_1(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """PAN c@1 metric (Peñas & Rodrigo, 2011).

    Scores exactly at threshold are treated as "unanswered" (preds == -1).
    Unanswered questions get partial credit equal to the accuracy on answered
    questions, rewarding a model that says "I don't know" over guessing wrong.

    Formula: c@1 = (n_c + acc_answered * n_u) / n
      where n_c = correct answers, n_u = unanswered, n = total.
    """
    n = len(labels)
    # 1 = authentic, 0 = fraud, -1 = abstain (score == threshold)
    preds = np.where(scores > threshold, 1, np.where(scores < threshold, 0, -1))
    answered = preds != -1
    n_u = int((~answered).sum())
    n_c = int((preds[answered] == labels[answered]).sum())
    if n == 0:
        return 0.0
    acc_answered = n_c / max(int(answered.sum()), 1)
    return float((n_c + acc_answered * n_u) / n)


def compute_f05u(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """PAN F0.5u metric (Bevendorff et al., 2019).

    F-beta with β=0.5 (precision-weighted) computed over the answered subset,
    with unanswered items counted as false negatives (missed authentics).
    This penalizes models that abstain too much.
    """
    preds = np.where(scores > threshold, 1, np.where(scores < threshold, 0, -1))
    answered = preds != -1

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    n_u = int((~answered).sum())  # unanswered items penalized as false negatives

    beta = 0.5  # β < 1 weights precision higher than recall
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn + n_u, 1)  # n_u added to denominator: abstaining hurts recall

    if precision + recall == 0:
        return 0.0
    return float((1 + beta**2) * precision * recall / (beta**2 * precision + recall))


def compute_pan_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute all four PAN verification metrics in one call."""
    return {
        "AUC": compute_auc(labels, scores),
        "EER": compute_eer(labels, scores),
        "c@1": compute_c_at_1(labels, scores),
        "F0.5u": compute_f05u(labels, scores),
    }