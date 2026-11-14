#!/usr/bin/env python3
"""Train and evaluate lightweight heads on frozen sentence features.

The module supports logistic regression, scikit-learn MLP, and PyTorch MLP
heads over ClinicalBERT embeddings or TF-IDF features. It writes predictions,
metrics, and optionally a reusable fitted head.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from omitehr.utils.evaluation import add_label_argument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize


def derive_embedding_tag(npz_path: str) -> str:
    """Pulls the distinguishing part out of an embeddings filename, e.g.
    'test_embeddings_discharge_bert.npz' -> 'embeddings_discharge_bert',
    so different embedding models stay distinguishable in the results
    file."""
    name = Path(npz_path).name
    if name.endswith(".npz"):
        name = name[:-len(".npz")]
    for prefix in ("train_", "test_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def load_labeled(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Loads embeddings/labels from a .npz, dropping any rows that
    aren't labeled yet."""
    data = np.load(npz_path, allow_pickle=False)
    if "labels" not in data.files:
        raise ValueError(
            f"{npz_path} has no 'labels' array -- regenerate the embeddings "
            f"from a labeled input JSONL so the companion NPZ includes labels."
        )
    X, y = data["embeddings"], data["labels"]
    mask = ~np.isnan(y)
    n_dropped = int((~mask).sum())
    if n_dropped:
        print(f"{npz_path.name}: dropping {n_dropped}/{len(y)} unlabeled sentences")
    if not mask.any():
        raise ValueError(
            f"{npz_path} has zero labeled sentences. Make sure the embeddings "
            f"were built from a labeled input JSONL, not an unlabeled split file."
        )
    return X[mask], y[mask].astype(int)


def load_labeled_text(jsonl_path: Path) -> Tuple[list, np.ndarray]:
    """Loads sentence text + labels from the embedding jsonl (the .npz
    doesn't have the text), dropping unlabeled rows.  Labels come from the
    companion NPZ when available because a resumed embedding run refreshes its
    labels from the latest labeling output, while existing JSONL rows may
    still contain an older label."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    sentences = [r["sentence"] for r in rows]
    npz_path = jsonl_path.with_suffix(".npz")
    if npz_path.exists():
        labels = np.load(npz_path, allow_pickle=False)["labels"]
        if len(rows) != len(labels):
            raise ValueError(
                f"JSONL/NPZ row mismatch for {jsonl_path}: "
                f"{len(rows)} JSONL rows vs {len(labels)} NPZ labels"
            )
    else:
        labels = np.array(
            [r["label"] if r["label"] is not None else np.nan for r in rows],
            dtype=np.float32,
        )
    mask = ~np.isnan(labels)
    n_dropped = int((~mask).sum())
    if n_dropped:
        print(f"{jsonl_path.name}: dropping {n_dropped}/{len(labels)} unlabeled sentences")
    if not mask.any():
        raise ValueError(f"{jsonl_path} has zero labeled sentences.")
    kept_sentences = [s for s, keep in zip(sentences, mask) if keep]
    return kept_sentences, labels[mask].astype(int)


def dedup_mask_within_doc(doc_ids, sentences) -> np.ndarray:
    """First-occurrence-per-admission-wins, case-insensitive exact-match
    mask -- True for rows to keep. Scoped to (doc_id, sentence text) so a
    repeated sentence across DIFFERENT admissions is untouched (still a
    genuinely different training example, paired with a different doc's
    other sentences/label distribution); only repeats WITHIN the same
    admission (e.g. an attestation line appearing 4x in one note) get
    collapsed to their first occurrence."""
    seen = set()
    mask = np.zeros(len(sentences), dtype=bool)
    for i, (doc_id, sentence) in enumerate(zip(doc_ids, sentences)):
        key = (str(doc_id), sentence.lower())
        if key not in seen:
            seen.add(key)
            mask[i] = True
    return mask


def load_sentence_metadata(jsonl_path: Path) -> Tuple[np.ndarray, np.ndarray, list]:
    """Loads doc_id/sentence_idx/sentence for every row in the embedding
    jsonl output, applying the same NaN-label mask load_labeled() uses on the
    companion NPZ.  The NPZ is the canonical label source after a resumed
    embedding run because its labels are refreshed from the latest labeling
    output even when existing JSONL rows retain older labels."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    npz_path = jsonl_path.with_suffix(".npz")
    if npz_path.exists():
        labels = np.load(npz_path, allow_pickle=False)["labels"]
        if len(rows) != len(labels):
            raise ValueError(
                f"JSONL/NPZ row mismatch for {jsonl_path}: "
                f"{len(rows)} JSONL rows vs {len(labels)} NPZ labels"
            )
    else:
        labels = np.array(
            [r["label"] if r["label"] is not None else np.nan for r in rows],
            dtype=np.float32,
        )
    mask = ~np.isnan(labels)
    doc_ids = np.array([r["doc_id"] for r in rows])[mask]
    sentence_idx = np.array([r["sentence_idx"] for r in rows])[mask]
    sentences = [s for s, keep in zip((r["sentence"] for r in rows), mask) if keep]
    return doc_ids, sentence_idx, sentences


def save_sentence_results(
    path: Path, doc_ids: np.ndarray, sentence_idx: np.ndarray, sentences: list,
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
) -> None:
    """Writes one row per test sentence, sorted by risk_rank (rank 1 --
    the model's single most likely omission -- first) -- doc_id,
    sentence text, true/predicted label, per-class probabilities, risk
    score, and that rank -- so you can just open this file and look
    top-to-bottom at exactly which sentences got flagged. Handles both
    a 3-class y_proba (the default) and a 2-class one (from
    --collapse_before_training)."""
    n_classes = y_proba.shape[1]
    risk_score = 1.0 - y_proba[:, 1:].sum(axis=1)  # P(not covered)
    order = np.argsort(-risk_score)  # order[0] = row index of rank 1, etc.
    rank = np.empty(len(risk_score), dtype=int)
    rank[order] = np.arange(1, len(risk_score) + 1)

    if n_classes == 2:
        proba_columns = ["proba_0_not_covered", "proba_1_covered"]
    else:
        proba_columns = ["proba_0_not_covered", "proba_1_partial", "proba_2_fully_covered"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "doc_id", "sentence_idx", "sentence", "true_label", "predicted_label",
            *proba_columns, "risk_score", "risk_rank",
        ])
        for i in order:
            writer.writerow([
                doc_ids[i], int(sentence_idx[i]), sentences[i],
                int(y_true[i]), int(y_pred[i]),
                *[float(y_proba[i, c]) for c in range(n_classes)],
                float(risk_score[i]), int(rank[i]),
            ])
    print(f"Saved sentence-level results ({len(sentences)} rows) -> {path}")


def compute_balanced_class_weights(y_train: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """sklearn's usual "balanced" weighting -- rarer classes get more
    weight."""
    counts = np.bincount(y_train, minlength=n_classes)
    return len(y_train) / (n_classes * np.maximum(counts, 1))


def oversample_to_balance(
    X_train: np.ndarray, y_train: np.ndarray, n_classes: int = 3, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oversamples every class up to the majority class's count, since
    MLPClassifier can't use class weights the way LogisticRegression
    can."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y_train, minlength=n_classes)
    target = int(counts.max())
    X_parts, y_parts = [], []
    for c in range(n_classes):
        idx = np.where(y_train == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=target, replace=True)
        X_parts.append(X_train[chosen])
        y_parts.append(y_train[chosen])
    # np.concatenate doesn't work on scipy sparse matrices (TF-IDF output).
    X_bal = sp.vstack(X_parts).tocsr() if sp.issparse(X_train) else np.concatenate(X_parts, axis=0)
    y_bal = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def fit_predict_logistic(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, max_iter: int = 1000,
    C: float = 1.0,
    pretrained_model: Optional[LogisticRegression] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LogisticRegression]:
    """Fits logistic regression, returning train + test predictions so
    the caller can compare them as an overfitting check, plus the
    fitted classifier itself (so the caller can save it). Lower C to
    fight overfitting on wide/concatenated feature sets.

    Pass an already-fit pretrained_model to skip fitting and just
    predict with it instead (e.g. after --load_model)."""
    if pretrained_model is not None:
        clf = pretrained_model
    else:
        clf = LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced")
        clf.fit(X_train, y_train)
    return (clf.predict(X_train), clf.predict_proba(X_train),
            clf.predict(X_test), clf.predict_proba(X_test), clf)


def fit_predict_sklearn_mlp(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    hidden_dim: int = 128, max_iter: int = 500, seed: int = 42,
    pretrained_model: Optional[MLPClassifier] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MLPClassifier]:
    """Oversamples to balance classes, then fits an MLP. Train
    predictions come from the original (non-oversampled) data so
    they're comparable to the test metrics. Also returns the fitted
    classifier so the caller can save it.

    Pass an already-fit pretrained_model to skip fitting and just
    predict with it instead (e.g. after --load_model)."""
    if pretrained_model is not None:
        clf = pretrained_model
    else:
        n_classes = int(y_train.max()) + 1
        X_bal, y_bal = oversample_to_balance(X_train, y_train, n_classes=n_classes, seed=seed)
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            max_iter=max_iter,
            early_stopping=True,
            random_state=seed,
        )
        clf.fit(X_bal, y_bal)
    return (clf.predict(X_train), clf.predict_proba(X_train),
            clf.predict(X_test), clf.predict_proba(X_test), clf)


def fit_predict_torch_mlp(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    hidden_dim: int = 128, epochs: int = 50, lr: float = 1e-3,
    batch_size: int = 64, seed: int = 42,
    pretrained_model=None,
):
    """A small Linear -> ReLU -> Dropout -> Linear head trained with
    Adam. Imports torch lazily so the sklearn-only path doesn't need
    it installed. Also returns the fitted model so the caller can save
    it.

    Pass an already-built pretrained_model to skip training and just
    predict with it instead (e.g. after --load_model)."""
    import torch
    import torch.nn as nn

    # Densify if this came from the --features tfidf path (sparse matrix).
    if sp.issparse(X_train):
        X_train = X_train.toarray()
    if sp.issparse(X_test):
        X_test = X_test.toarray()

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = int(y_train.max()) + 1
    in_dim = X_train.shape[1]

    if pretrained_model is not None:
        model = pretrained_model.to(device)
    else:
        model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        ).to(device)

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)

        # Same balanced reweighting as LogisticRegression's class_weight.
        class_weights = compute_balanced_class_weights(y_train, n_classes=n_classes)
        weight_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(weight=weight_t)

        n = X_train_t.shape[0]
        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = X_train_t[idx], y_train_t[idx]
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
            if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
                print(f"  epoch {epoch + 1}/{epochs}: loss={total_loss / n:.4f}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        train_logits = model(X_train_t)
        train_proba = torch.softmax(train_logits, dim=1).cpu().numpy()
        train_preds = train_logits.argmax(dim=1).cpu().numpy()
        logits = model(X_test_t)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    return train_preds, train_proba, preds, proba, model


def save_classifier(clf, classifier: str, path: Path) -> None:
    """Saves a fitted classifier so a later run can reload it with
    --load_model instead of refitting. logistic/mlp use joblib;
    torch_mlp saves the weights plus the shape info needed to rebuild
    the same architecture before reloading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if classifier == "torch_mlp":
        import torch
        torch.save({
            "state_dict": clf.state_dict(),
            "in_dim": clf[0].in_features,
            "hidden_dim": clf[0].out_features,
            "n_classes": clf[-1].out_features,
        }, path)
    else:
        import joblib
        joblib.dump(clf, path)
    print(f"Saved fitted model -> {path}")


def load_classifier(classifier: str, path: Path):
    """Loads a classifier saved by save_classifier(). For torch_mlp,
    rebuilds the same architecture from the saved shape info before
    loading the weights."""
    if classifier == "torch_mlp":
        import torch
        import torch.nn as nn
        ckpt = torch.load(path, map_location="cpu")
        model = nn.Sequential(
            nn.Linear(ckpt["in_dim"], ckpt["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ckpt["hidden_dim"], ckpt["n_classes"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model
    else:
        import joblib
        return joblib.load(path)


def safe_auc_metrics_multiclass(
    y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 3,
) -> Tuple[Optional[float], Optional[float]]:
    """macro one-vs-rest AUROC/AUPRC. Returns None for whichever metric
    can't be computed (e.g. a class missing from y_true)."""
    try:
        auroc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError as e:
        print(f"  (3-class AUROC unavailable: {e})")
        auroc = None
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        auprc = float(average_precision_score(y_true_bin, y_proba, average="macro"))
    except ValueError as e:
        print(f"  (3-class AUPRC unavailable: {e})")
        auprc = None
    return auroc, auprc


def safe_auc_metrics_binary(
    y_true_bin: np.ndarray, y_score_bin: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    try:
        auroc = float(roc_auc_score(y_true_bin, y_score_bin))
    except ValueError as e:
        print(f"  (binary AUROC unavailable: {e})")
        auroc = None
    try:
        auprc = float(average_precision_score(y_true_bin, y_score_bin))
    except ValueError as e:
        print(f"  (binary AUPRC unavailable: {e})")
        auprc = None
    return auroc, auprc


def recall_at_k(
    y_test_bin: np.ndarray, y_score_bin: np.ndarray, k: int, positive_class: int = 0,
) -> Optional[Tuple[float, int, int]]:
    """Screening recall@k: rank test sentences by predicted risk of
    being an omission (1 - P(covered)), take the k highest-risk ones,
    and report what fraction of the true omissions (y_test_bin == 0)
    are caught in that top-k. Answers "if we can only manually review
    k sentences, how many real omissions do we catch?" None if there
    are no true omissions to recall or k is non-positive.

    positive_class=0 (default) is the screening framing above. Pass
    positive_class=1 to flip it around: rank by P(covered) directly
    and ask how many of the true "covered" sentences land in the
    top-k instead -- e.g. for checking which sentences the model is
    most confidently right about, rather than screening for misses.

    Returns (recall, n_caught, n_positive) -- callers should print the
    raw counts too, not just the ratio: with a large n_positive (e.g.
    tens of thousands), catching just 1-2 already rounds to 0.000 at
    typical print precision, which reads as "caught nothing" even
    though it isn't."""
    n_positive = int((y_test_bin == positive_class).sum())
    if n_positive == 0 or k <= 0:
        return None
    score = y_score_bin if positive_class == 1 else 1.0 - y_score_bin
    top_k = np.argsort(-score)[:k]
    n_caught = int((y_test_bin[top_k] == positive_class).sum())
    return n_caught / n_positive, n_caught, n_positive


def per_admission_auc_metrics(
    doc_ids: np.ndarray, y_test_bin: np.ndarray, y_score_bin: np.ndarray,
) -> Optional[Tuple[float, float, int, int]]:
    """Per-admission counterpart to safe_auc_metrics_binary: computes binary
    AUROC/AUPRC separately within each admission (doc_id) and averages,
    instead of pooling every sentence in the test set into one ranking --
    same motivation as per_admission_recall_at_k (a typical admission's own
    separation, not a global ranking that mixes sentences across different
    patients' notes together). Admissions missing one class entirely (all
    covered, or all not-covered) are skipped -- AUROC/AUPRC aren't defined
    without both classes present in that admission.

    Returns (mean_auroc, mean_auprc, n_admissions_used, n_admissions_skipped),
    or None if no admission has both classes present."""
    aurocs = []
    auprcs = []
    n_skipped = 0
    for doc_id in np.unique(doc_ids):
        mask = doc_ids == doc_id
        local_bin = y_test_bin[mask]
        if local_bin.sum() == 0 or local_bin.sum() == len(local_bin):
            n_skipped += 1
            continue  # need both classes present to compute AUROC/AUPRC
        local_score = y_score_bin[mask]
        aurocs.append(roc_auc_score(local_bin, local_score))
        auprcs.append(average_precision_score(local_bin, local_score))
    if not aurocs:
        return None
    return float(np.mean(aurocs)), float(np.mean(auprcs)), len(aurocs), n_skipped


def per_admission_recall_at_k(
    doc_ids: np.ndarray, y_test_bin: np.ndarray, y_score_bin: np.ndarray, k: int,
    positive_class: int = 0,
) -> Optional[Tuple[float, int]]:
    """Same idea as recall_at_k, but computed separately within each
    admission (doc_id) and then averaged, instead of pooling every
    sentence in the test set into one ranking. The pooled version can
    be dominated by whichever admissions happen to have the most
    sentences; this asks "for a typical admission, if we review its
    top k highest-risk sentences, what fraction of its omissions do
    we catch?" (or, with positive_class=1, "...what fraction of its
    covered sentences do we catch?")

    Admissions with fewer than k sentences use all of their sentences
    (so recall can still hit 1.0 for a short admission). Admissions
    with zero true instances of positive_class are skipped -- there's
    nothing to recall.

    Returns (mean_recall, n_admissions_used), or None if no admission
    has any true instances of positive_class."""
    if k <= 0:
        return None
    per_admission_recall = []
    for doc_id in np.unique(doc_ids):
        mask = doc_ids == doc_id
        local_bin = y_test_bin[mask]
        n_local_positive = int((local_bin == positive_class).sum())
        if n_local_positive == 0:
            continue
        local_score = y_score_bin[mask]
        local_k = min(k, mask.sum())
        score = local_score if positive_class == 1 else 1.0 - local_score
        top_k = np.argsort(-score)[:local_k]
        n_caught = int((local_bin[top_k] == positive_class).sum())
        per_admission_recall.append(n_caught / n_local_positive)
    if not per_admission_recall:
        return None
    return float(np.mean(per_admission_recall)), len(per_admission_recall)


def recall_at_k_baselines(
    y_test_bin: np.ndarray, k: int, positive_class: int = 0,
) -> Optional[Tuple[float, float]]:
    """Two reference points for recall_at_k, so a model's recall@k can
    be read as "better than chance by how much" instead of in a vacuum:

    - random: expected recall@k if the k reviewed sentences were picked
      at random instead of by model score (k / n_total, capped at 1).
    - ceiling: the best any ranking could do -- what recall@k would be
      if every true positive were ranked first (k / n_positive, capped
      at 1). Below this, k is the bottleneck, not the model.

    None if there's nothing to recall or k is non-positive."""
    n_total = len(y_test_bin)
    n_positive = int((y_test_bin == positive_class).sum())
    if n_positive == 0 or k <= 0 or n_total == 0:
        return None
    random_recall = min(1.0, k / n_total)
    ceiling_recall = min(1.0, k / n_positive)
    return random_recall, ceiling_recall


def per_admission_recall_at_k_baselines(
    doc_ids: np.ndarray, y_test_bin: np.ndarray, k: int, positive_class: int = 0,
) -> Optional[Tuple[float, float, int]]:
    """Per-admission counterpart to recall_at_k_baselines, averaged the
    same way and over the same set of admissions as
    per_admission_recall_at_k, so the two are directly comparable.

    Returns (mean_random_recall, mean_ceiling_recall, n_admissions_used),
    or None if no admission has any true instances of positive_class."""
    if k <= 0:
        return None
    random_vals = []
    ceiling_vals = []
    for doc_id in np.unique(doc_ids):
        mask = doc_ids == doc_id
        n_local_total = int(mask.sum())
        n_local_positive = int((y_test_bin[mask] == positive_class).sum())
        if n_local_positive == 0:
            continue
        random_vals.append(min(1.0, k / n_local_total))
        ceiling_vals.append(min(1.0, k / n_local_positive))
    if not random_vals:
        return None
    return float(np.mean(random_vals)), float(np.mean(ceiling_vals)), len(random_vals)


def precision_at_k(
    y_test_bin: np.ndarray, y_score_bin: np.ndarray, k: int, positive_class: int = 0,
) -> Optional[Tuple[float, int, int]]:
    """Precision@k: of the k sentences ranked highest (by omission risk
    if positive_class=0, or by P(covered) directly if positive_class=1),
    what fraction actually are positive_class? Answers "if we hand a
    reviewer k flagged sentences, how many are worth their time?" --
    the flip side of recall@k, which asks how much of the positive
    class those k sentences cover.

    Unlike recall@k, this is defined even when there are zero true
    positives (precision is just 0 then) -- only None when k <= 0 or
    the test set is empty. k is capped at the test set size if larger.

    Returns (precision, n_caught, k_used)."""
    n_total = len(y_test_bin)
    if k <= 0 or n_total == 0:
        return None
    k_used = min(k, n_total)
    score = y_score_bin if positive_class == 1 else 1.0 - y_score_bin
    top_k = np.argsort(-score)[:k_used]
    n_caught = int((y_test_bin[top_k] == positive_class).sum())
    return n_caught / k_used, n_caught, k_used


def per_admission_precision_at_k(
    doc_ids: np.ndarray, y_test_bin: np.ndarray, y_score_bin: np.ndarray, k: int,
    positive_class: int = 0,
) -> Optional[Tuple[float, int]]:
    """Per-admission counterpart to precision_at_k: ranks each
    admission's own sentences separately and averages the precision@k
    across every admission (unlike per_admission_recall_at_k, there's
    no reason to skip admissions with zero positives here -- precision
    is still meaningful, it's just 0 for that admission).

    Returns (mean_precision, n_admissions_used), or None if k <= 0 or
    doc_ids is empty."""
    if k <= 0 or len(doc_ids) == 0:
        return None
    per_admission_precision = []
    for doc_id in np.unique(doc_ids):
        mask = doc_ids == doc_id
        local_bin = y_test_bin[mask]
        local_score = y_score_bin[mask]
        local_k = min(k, mask.sum())
        score = local_score if positive_class == 1 else 1.0 - local_score
        top_k = np.argsort(-score)[:local_k]
        n_caught = int((local_bin[top_k] == positive_class).sum())
        per_admission_precision.append(n_caught / local_k)
    return float(np.mean(per_admission_precision)), len(per_admission_precision)


def precision_at_k_baselines(
    y_test_bin: np.ndarray, k: int, positive_class: int = 0,
) -> Optional[Tuple[float, float]]:
    """Reference points for precision_at_k:

    - random: expected precision@k if the k sentences were picked at
      random -- this is just the overall prevalence of positive_class
      (n_positive / n_total), independent of k.
    - ceiling: the best any ranking could do at this k -- if k is
      smaller than n_positive, a perfect ranking fills all k slots
      with positives (precision 1); otherwise it's capped by how many
      positives actually exist (n_positive / k).

    None if k <= 0 or the test set is empty."""
    n_total = len(y_test_bin)
    if k <= 0 or n_total == 0:
        return None
    n_positive = int((y_test_bin == positive_class).sum())
    random_precision = n_positive / n_total
    # k_used, not k -- if k > n_total, the ceiling must fall back to
    # n_positive/n_total (forced by including every example), matching
    # per_admission_precision_at_k_baselines' local_k handling below.
    k_used = min(k, n_total)
    ceiling_precision = min(1.0, n_positive / k_used)
    return random_precision, ceiling_precision


def per_admission_precision_at_k_baselines(
    doc_ids: np.ndarray, y_test_bin: np.ndarray, k: int, positive_class: int = 0,
) -> Optional[Tuple[float, float, int]]:
    """Per-admission counterpart to precision_at_k_baselines, averaged
    the same way and over the same admissions as
    per_admission_precision_at_k, so the two are directly comparable.

    Returns (mean_random_precision, mean_ceiling_precision, n_admissions_used),
    or None if k <= 0 or doc_ids is empty."""
    if k <= 0 or len(doc_ids) == 0:
        return None
    random_vals = []
    ceiling_vals = []
    for doc_id in np.unique(doc_ids):
        mask = doc_ids == doc_id
        n_local_total = int(mask.sum())
        n_local_positive = int((y_test_bin[mask] == positive_class).sum())
        local_k = min(k, n_local_total)
        random_vals.append(n_local_positive / n_local_total)
        ceiling_vals.append(min(1.0, n_local_positive / local_k))
    return float(np.mean(random_vals)), float(np.mean(ceiling_vals)), len(random_vals)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-encoder classifier: encoder embedding -> coverage label."
    )
    add_label_argument(parser)
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--classifier", type=str, default="logistic",
                        choices=["logistic", "mlp", "torch_mlp"],
                        help="logistic (sklearn, default) | mlp (sklearn "
                            "MLPClassifier) | torch_mlp (custom torch head)")
    parser.add_argument("--features", type=str, default="bert",
                        choices=["bert", "tfidf"],
                        help="bert (default): the .npz embeddings. "
                            "tfidf: TfidfVectorizer on sentence text from the "
                            "matching .jsonl (same base name as --train_npz/--test_npz)")
    parser.add_argument("--tfidf_max_features", type=int, default=2000,
                        help="Vocabulary size cap for --features tfidf")
    parser.add_argument("--collapse_before_training", action="store_true",
                        help="Collapse labels to binary (0=not covered, 1=covered) "
                            "before fitting, training a genuinely binary classifier, "
                            "instead of training on the 3-class label and collapsing "
                            "predictions afterward. Only the binary section gets "
                            "reported in this mode.")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Max iterations for logistic/mlp")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse L2 regularization strength for --classifier logistic. "
                            "Lower to fight overfitting on high-dim/concatenated features. "
                            "Ignored for mlp/torch_mlp.")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden layer size for mlp/torch_mlp")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs for torch_mlp")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for torch_mlp")
    parser.add_argument("--results_json", type=str, default=None,
                        help="Append this run's results into the JSON array "
                            "stored here (default: classify_results.json next "
                            "to --test_npz)")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Also save y_test/y_proba to a companion "
                            "predictions_<classifier>_<tag>.npz (next to --results_json) "
                            "for plotting ROC/PR curves later.")
    parser.add_argument("--save_sentence_results", action="store_true",
                        help="Also save a per-sentence CSV (doc_id, sentence text, "
                            "true/predicted label, per-class probabilities, risk rank) "
                            "to sentence_results_<classifier>_<tag>.csv next to "
                            "--results_json, so you can open it and look sentence by "
                            "sentence at what got flagged.")
    parser.add_argument("--recall_k", type=str, default="5,10",
                        help="Comma-separated k's for screening recall@k: of the k "
                            "test sentences ranked highest-risk for omission, what "
                            "fraction of the true omissions get caught.")
    parser.add_argument("--recall_positive_class", type=int, default=0, choices=[0, 1],
                        help="Which binary label recall@k targets: 0 (default) is the "
                            "screening framing -- rank by omission risk, catch true "
                            "omissions. 1 flips it -- rank by P(covered) directly, "
                            "catch true covered sentences (e.g. to check which "
                            "sentences the model is most confidently right about).")
    parser.add_argument("--dedup_train", action="store_true",
                        help="Drop repeated (case-insensitive exact match) sentences "
                            "WITHIN the same admission from --train_npz before fitting "
                            "(e.g. an attestation line appearing several times in one "
                            "note) -- doesn't touch --test_npz, and doesn't dedup a "
                            "sentence repeating across DIFFERENT admissions (still a "
                            "genuinely different training example there).")
    parser.add_argument("--save_model", type=str, default=None,
                        help="Save the fitted classifier here after training, so a "
                            "later run can skip refitting via --load_model.")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Load a previously --save_model'd classifier instead of "
                            "fitting a new one on --train_npz -- must match --classifier. "
                            "Still predicts/evaluates on both --train_npz and --test_npz.")
    args = parser.parse_args()
    if args.positive_label == 2:
        args.collapse_before_training = True
    recall_k_values = [int(k.strip()) for k in args.recall_k.split(",") if k.strip()]

    if args.features == "bert":
        X_train, y_train = load_labeled(Path(args.train_npz))
        X_test, y_test = load_labeled(Path(args.test_npz))
        if args.dedup_train:
            train_doc_ids, _, train_sentences = load_sentence_metadata(
                Path(args.train_npz).with_suffix(".jsonl"))
            mask = dedup_mask_within_doc(train_doc_ids, train_sentences)
            print(f"--dedup_train: dropped {len(mask) - int(mask.sum())}/{len(mask)} "
                  f"within-admission repeated sentences from the train set")
            X_train, y_train = X_train[mask], y_train[mask]
    else:
        train_jsonl = Path(args.train_npz).with_suffix(".jsonl")
        test_jsonl = Path(args.test_npz).with_suffix(".jsonl")
        train_sentences, y_train = load_labeled_text(train_jsonl)
        test_sentences, y_test = load_labeled_text(test_jsonl)
        if args.dedup_train:
            train_doc_ids, _, _ = load_sentence_metadata(train_jsonl)
            mask = dedup_mask_within_doc(train_doc_ids, train_sentences)
            print(f"--dedup_train: dropped {len(mask) - int(mask.sum())}/{len(mask)} "
                  f"within-admission repeated sentences from the train set")
            train_sentences = [s for s, keep in zip(train_sentences, mask) if keep]
            y_train = y_train[mask]
        vectorizer = TfidfVectorizer(max_features=args.tfidf_max_features)
        X_train = vectorizer.fit_transform(train_sentences)
        X_test = vectorizer.transform(test_sentences)
        print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    if args.collapse_before_training:
        y_train = (y_train >= args.positive_label).astype(int)
        y_test = (y_test >= args.positive_label).astype(int)
        print(f"Collapsed labels to binary with positive label >= {args.positive_label}.")

    def label_distribution(y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        total = len(y)
        return {
            str(label): {"count": count, "pct": round(100 * count / total, 1)}
            for label, count in counts.items()
        }

    train_dist = label_distribution(y_train)
    test_dist = label_distribution(y_test)
    train_counts = {k: v["count"] for k, v in train_dist.items()}
    test_counts = {k: v["count"] for k, v in test_dist.items()}

    print("\n=== Label distribution ===")
    print(f"Train ({len(y_train)} labeled): " +
          ", ".join(f"{label}={v['count']} ({v['pct']}%)" for label, v in sorted(train_dist.items())))
    print(f"Test  ({len(y_test)} labeled): " +
          ", ".join(f"{label}={v['count']} ({v['pct']}%)" for label, v in sorted(test_dist.items())))

    pretrained_model = None
    if args.load_model:
        pretrained_model = load_classifier(args.classifier, Path(args.load_model))
        print(f"Loaded pretrained --classifier {args.classifier} from "
              f"{args.load_model} (skipping fit)")
    else:
        print(f"\nFitting --classifier {args.classifier}...")

    if args.classifier == "logistic":
        y_train_pred, y_train_proba, y_pred, y_proba, clf = fit_predict_logistic(
            X_train, y_train, X_test, max_iter=args.max_iter, C=args.C,
            pretrained_model=pretrained_model)
    elif args.classifier == "mlp":
        y_train_pred, y_train_proba, y_pred, y_proba, clf = fit_predict_sklearn_mlp(
            X_train, y_train, X_test, hidden_dim=args.hidden_dim, max_iter=args.max_iter,
            pretrained_model=pretrained_model)
    else:
        y_train_pred, y_train_proba, y_pred, y_proba, clf = fit_predict_torch_mlp(
            X_train, y_train, X_test, hidden_dim=args.hidden_dim,
            epochs=args.epochs, lr=args.lr, pretrained_model=pretrained_model)

    if args.save_model:
        save_classifier(clf, args.classifier, Path(args.save_model))

    three_class_acc = three_class_cm = three_class_auroc = three_class_auprc = None
    train_acc = train_auroc = train_auprc = None

    if not args.collapse_before_training:
        three_class_acc = accuracy_score(y_test, y_pred)
        three_class_cm = confusion_matrix(y_test, y_pred)
        three_class_auroc, three_class_auprc = safe_auc_metrics_multiclass(y_test, y_proba)

        # Same metrics on the training set, as an overfitting check -- a
        # big train-vs-test gap matters more than the raw train number.
        train_acc = accuracy_score(y_train, y_train_pred)
        train_auroc, train_auprc = safe_auc_metrics_multiclass(y_train, y_train_proba)
        print("\n=== Train stats (overfitting check) ===")
        print(f"Train accuracy: {train_acc:.3f}  (test: {three_class_acc:.3f}, "
              f"gap: {train_acc - three_class_acc:+.3f})")
        if train_auroc is not None and three_class_auroc is not None:
            print(f"Train AUROC (macro, one-vs-rest): {train_auroc:.3f}  "
                  f"(test: {three_class_auroc:.3f}, gap: {train_auroc - three_class_auroc:+.3f})")
        else:
            print(f"Train AUROC (macro, one-vs-rest): "
                  f"{'N/A' if train_auroc is None else f'{train_auroc:.3f}'}")
        if train_auprc is not None and three_class_auprc is not None:
            print(f"Train AUPRC (macro, one-vs-rest): {train_auprc:.3f}  "
                  f"(test: {three_class_auprc:.3f}, gap: {train_auprc - three_class_auprc:+.3f})")
        else:
            print(f"Train AUPRC (macro, one-vs-rest): "
                  f"{'N/A' if train_auprc is None else f'{train_auprc:.3f}'}")

        print("\n=== 3-class (0=not covered, 1=partially, 2=fully covered) ===")
        print(f"Accuracy: {three_class_acc:.3f}")
        print(f"AUROC (macro, one-vs-rest): {three_class_auroc:.3f}"
              if three_class_auroc is not None else "AUROC (macro, one-vs-rest): N/A")
        print(f"AUPRC (macro, one-vs-rest): {three_class_auprc:.3f}"
              if three_class_auprc is not None else "AUPRC (macro, one-vs-rest): N/A")
        print(classification_report(y_test, y_pred, digits=3, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(three_class_cm)
    else:
        print("\n(--collapse_before_training set -- skipping the 3-class section, "
              "since the model was trained directly on the binary label.)")

    # Binary collapse: covered (partial or full) vs not -- often the
    # more actionable framing. P(covered) = P(label=1) + P(label=2).
    y_test_bin = (y_test >= 1).astype(int)
    y_pred_bin = (y_pred >= 1).astype(int)
    y_score_bin = y_proba[:, 1:].sum(axis=1)
    binary_acc = accuracy_score(y_test_bin, y_pred_bin)
    binary_cm = confusion_matrix(y_test_bin, y_pred_bin)
    binary_auroc, binary_auprc = safe_auc_metrics_binary(y_test_bin, y_score_bin)
    print("\n=== Binary (0=not covered, 1=covered [partial or full]) ===")
    print(f"Accuracy: {binary_acc:.3f}")
    print(f"AUROC: {binary_auroc:.3f}" if binary_auroc is not None else "AUROC: N/A")
    print(f"AUPRC: {binary_auprc:.3f}" if binary_auprc is not None else "AUPRC: N/A")
    print(classification_report(y_test_bin, y_pred_bin, digits=3, zero_division=0,
                                target_names=["not covered", "covered"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(binary_cm)

    # Loaded here (not just inside --save_sentence_results below) since
    # per-admission recall@k needs doc_id to group sentences by admission.
    test_jsonl_path = Path(args.test_npz).with_suffix(".jsonl")
    doc_ids, sentence_idx, sentences = load_sentence_metadata(test_jsonl_path)

    # Secondary metric alongside the pooled binary AUROC/AUPRC above --
    # computed per-admission then averaged, so sentences from different
    # patients' notes are never ranked directly against each other.
    per_admission_auc = per_admission_auc_metrics(doc_ids, y_test_bin, y_score_bin)
    if per_admission_auc is not None:
        pa_auroc, pa_auprc, pa_n_used, pa_n_skipped = per_admission_auc
        print(f"\nPer-admission AUROC: {pa_auroc:.3f}  (avg over {pa_n_used} admissions"
              f"{f', {pa_n_skipped} skipped -- missing a class' if pa_n_skipped else ''})")
        print(f"Per-admission AUPRC: {pa_auprc:.3f}")
    else:
        pa_auroc = pa_auprc = pa_n_used = pa_n_skipped = None
        print("\nPer-admission AUROC/AUPRC: unavailable (no admission has both classes present)")

    pos_class = args.recall_positive_class
    pos_label = "omissions" if pos_class == 0 else "covered sentences"
    pos_label_singular = "omission" if pos_class == 0 else "covered sentence"

    recall_at_k_results: Dict[str, Optional[float]] = {}
    recall_at_k_random_results: Dict[str, Optional[float]] = {}
    recall_at_k_ceiling_results: Dict[str, Optional[float]] = {}
    n_positive_total = int((y_test_bin == pos_class).sum())
    print(f"\nScreening recall@k ({n_positive_total} true {pos_label} in test set):")
    for k in recall_k_values:
        result_k = recall_at_k(y_test_bin, y_score_bin, k, positive_class=pos_class)
        baseline_k = recall_at_k_baselines(y_test_bin, k, positive_class=pos_class)
        random_recall, ceiling_recall = baseline_k if baseline_k is not None else (None, None)
        recall_at_k_random_results[str(k)] = random_recall
        recall_at_k_ceiling_results[str(k)] = ceiling_recall
        if result_k is None:
            recall_at_k_results[str(k)] = None
            print(f"  recall@{k}: N/A")
        else:
            r_at_k, n_caught, _ = result_k
            recall_at_k_results[str(k)] = r_at_k
            # Raw counts shown alongside the ratio -- with a huge
            # n_positive_total, catching 1-2 rounds to 0.000 and reads
            # as "caught nothing" even when it isn't.
            print(f"  recall@{k}: {r_at_k:.5f}  ({n_caught}/{n_positive_total} caught)  "
                  f"[random: {random_recall:.5f}, ceiling: {ceiling_recall:.5f}]")

    per_admission_recall_at_k_results: Dict[str, Optional[float]] = {}
    per_admission_recall_at_k_random_results: Dict[str, Optional[float]] = {}
    per_admission_recall_at_k_ceiling_results: Dict[str, Optional[float]] = {}
    print(f"\nScreening recall@k, per-admission average (recall computed within each "
          f"doc_id separately, then averaged across admissions with >=1 true "
          f"{pos_label_singular}):")
    for k in recall_k_values:
        result_k = per_admission_recall_at_k(doc_ids, y_test_bin, y_score_bin, k,
                                              positive_class=pos_class)
        baseline_k = per_admission_recall_at_k_baselines(doc_ids, y_test_bin, k,
                                                          positive_class=pos_class)
        if baseline_k is not None:
            random_recall, ceiling_recall, _ = baseline_k
        else:
            random_recall = ceiling_recall = None
        per_admission_recall_at_k_random_results[str(k)] = random_recall
        per_admission_recall_at_k_ceiling_results[str(k)] = ceiling_recall
        if result_k is None:
            per_admission_recall_at_k_results[str(k)] = None
            print(f"  recall@{k} (per-admission): N/A")
        else:
            r_at_k, n_admissions_used = result_k
            per_admission_recall_at_k_results[str(k)] = r_at_k
            print(f"  recall@{k} (per-admission): {r_at_k:.5f}  "
                  f"(averaged over {n_admissions_used} admissions)  "
                  f"[random: {random_recall:.5f}, ceiling: {ceiling_recall:.5f}]")

    precision_at_k_results: Dict[str, Optional[float]] = {}
    precision_at_k_random_results: Dict[str, Optional[float]] = {}
    precision_at_k_ceiling_results: Dict[str, Optional[float]] = {}
    print(f"\nScreening precision@k (of the k sentences flagged as highest-risk "
          f"{pos_label_singular}, what fraction actually are):")
    for k in recall_k_values:
        result_k = precision_at_k(y_test_bin, y_score_bin, k, positive_class=pos_class)
        baseline_k = precision_at_k_baselines(y_test_bin, k, positive_class=pos_class)
        if baseline_k is not None:
            random_precision, ceiling_precision = baseline_k
        else:
            random_precision = ceiling_precision = None
        precision_at_k_random_results[str(k)] = random_precision
        precision_at_k_ceiling_results[str(k)] = ceiling_precision
        if result_k is None:
            precision_at_k_results[str(k)] = None
            print(f"  precision@{k}: N/A")
        else:
            p_at_k, n_caught, k_used = result_k
            precision_at_k_results[str(k)] = p_at_k
            print(f"  precision@{k}: {p_at_k:.5f}  ({n_caught}/{k_used} caught)  "
                  f"[random: {random_precision:.5f}, ceiling: {ceiling_precision:.5f}]")

    per_admission_precision_at_k_results: Dict[str, Optional[float]] = {}
    per_admission_precision_at_k_random_results: Dict[str, Optional[float]] = {}
    per_admission_precision_at_k_ceiling_results: Dict[str, Optional[float]] = {}
    print(f"\nScreening precision@k, per-admission average (computed within each "
          f"doc_id separately, then averaged across all admissions):")
    for k in recall_k_values:
        result_k = per_admission_precision_at_k(doc_ids, y_test_bin, y_score_bin, k,
                                                 positive_class=pos_class)
        baseline_k = per_admission_precision_at_k_baselines(doc_ids, y_test_bin, k,
                                                              positive_class=pos_class)
        if baseline_k is not None:
            random_precision, ceiling_precision, _ = baseline_k
        else:
            random_precision = ceiling_precision = None
        per_admission_precision_at_k_random_results[str(k)] = random_precision
        per_admission_precision_at_k_ceiling_results[str(k)] = ceiling_precision
        if result_k is None:
            per_admission_precision_at_k_results[str(k)] = None
            print(f"  precision@{k} (per-admission): N/A")
        else:
            p_at_k, n_admissions_used = result_k
            per_admission_precision_at_k_results[str(k)] = p_at_k
            print(f"  precision@{k} (per-admission): {p_at_k:.5f}  "
                  f"(averaged over {n_admissions_used} admissions)  "
                  f"[random: {random_precision:.5f}, ceiling: {ceiling_precision:.5f}]")

    # Binary counterpart to the 3-class overfitting check above.
    y_train_bin = (y_train >= 1).astype(int)
    y_train_pred_bin = (y_train_pred >= 1).astype(int)
    y_train_score_bin = y_train_proba[:, 1:].sum(axis=1)
    train_binary_acc = accuracy_score(y_train_bin, y_train_pred_bin)
    train_binary_auroc, train_binary_auprc = safe_auc_metrics_binary(y_train_bin, y_train_score_bin)
    print("\n=== Train stats, binary (overfitting check) ===")
    print(f"Train accuracy: {train_binary_acc:.3f}  (test: {binary_acc:.3f}, "
          f"gap: {train_binary_acc - binary_acc:+.3f})")
    if train_binary_auroc is not None and binary_auroc is not None:
        print(f"Train AUROC: {train_binary_auroc:.3f}  "
              f"(test: {binary_auroc:.3f}, gap: {train_binary_auroc - binary_auroc:+.3f})")
    else:
        print(f"Train AUROC: {'N/A' if train_binary_auroc is None else f'{train_binary_auroc:.3f}'}")
    if train_binary_auprc is not None and binary_auprc is not None:
        print(f"Train AUPRC: {train_binary_auprc:.3f}  "
              f"(test: {binary_auprc:.3f}, gap: {train_binary_auprc - binary_auprc:+.3f})")
    else:
        print(f"Train AUPRC: {'N/A' if train_binary_auprc is None else f'{train_binary_auprc:.3f}'}")

    if args.results_json:
        results_path = Path(args.results_json)
    else:
        # Collapsed-binary runs get their own subfolder by default, so they
        # never mix with the regular 3-class run's results/predictions/
        # sentence-results files in the same directory.
        results_dir = Path(args.test_npz).parent
        if args.collapse_before_training:
            results_dir = results_dir / "collapsed_binary"
        results_path = results_dir / "classify_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    tag = args.features if args.features == "tfidf" else derive_embedding_tag(args.test_npz)

    if args.save_predictions:
        pred_path = results_path.parent / f"predictions_{args.classifier}_{tag}.npz"
        np.savez(pred_path, y_test=y_test, y_proba=y_proba)
        print(f"Saved predictions -> {pred_path}")

    if args.save_sentence_results:
        # doc_ids/sentence_idx/sentences already loaded above for per-admission recall@k.
        sentence_results_path = results_path.parent / f"sentence_results_{args.classifier}_{tag}.csv"
        save_sentence_results(sentence_results_path, doc_ids, sentence_idx, sentences,
                               y_test, y_pred, y_proba)

    three_class_diagnostics = None
    three_class_result = None
    if not args.collapse_before_training:
        three_class_diagnostics = {
            "accuracy": float(train_acc),
            "auroc_macro_ovr": train_auroc,
            "auprc_macro_ovr": train_auprc,
            "accuracy_gap": float(train_acc - three_class_acc),
            "auroc_gap": (train_auroc - three_class_auroc)
                if train_auroc is not None and three_class_auroc is not None else None,
            "auprc_gap": (train_auprc - three_class_auprc)
                if train_auprc is not None and three_class_auprc is not None else None,
        }
        three_class_result = {
            "accuracy": float(three_class_acc),
            "auroc_macro_ovr": three_class_auroc,
            "auprc_macro_ovr": three_class_auprc,
            "classification_report": classification_report(
                y_test, y_pred, digits=3, zero_division=0, output_dict=True),
            "confusion_matrix": three_class_cm.tolist(),
        }

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "classifier": args.classifier,
        "features": args.features,
        "collapse_before_training": args.collapse_before_training,
        "positive_label": args.positive_label,
        "embedding_source": derive_embedding_tag(args.test_npz) if args.features == "bert" else None,
        "hyperparams": {
            "max_iter": args.max_iter,
            "C": args.C if args.classifier == "logistic" else None,
            "hidden_dim": args.hidden_dim if args.classifier != "logistic" else None,
            "epochs": args.epochs if args.classifier == "torch_mlp" else None,
            "lr": args.lr if args.classifier == "torch_mlp" else None,
            "tfidf_max_features": args.tfidf_max_features if args.features == "tfidf" else None,
        },
        "train_npz": str(args.train_npz),
        "test_npz": str(args.test_npz),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_label_distribution": train_dist,
        "test_label_distribution": test_dist,
        "train_diagnostics": {
            # Same metrics as "three_class"/"binary" below, evaluated on
            # the training set. No confusion matrix/classification_report
            # here to keep it cheap -- accuracy/AUROC/AUPRC are enough to
            # spot a gap worth investigating. "three_class" is None when
            # --collapse_before_training was used.
            "three_class": three_class_diagnostics,
            "binary": {
                "accuracy": float(train_binary_acc),
                "auroc": train_binary_auroc,
                "auprc": train_binary_auprc,
                "accuracy_gap": float(train_binary_acc - binary_acc),
                "auroc_gap": (train_binary_auroc - binary_auroc)
                    if train_binary_auroc is not None and binary_auroc is not None else None,
                "auprc_gap": (train_binary_auprc - binary_auprc)
                    if train_binary_auprc is not None and binary_auprc is not None else None,
            },
        },
        # None when --collapse_before_training was used -- no 3-class
        # distinction left to report.
        "three_class": three_class_result,
        "binary": {
            "accuracy": float(binary_acc),
            "auroc": binary_auroc,
            "auprc": binary_auprc,
            "auroc_per_admission": pa_auroc,
            "auprc_per_admission": pa_auprc,
            "n_admissions_used_for_auc": pa_n_used,
            "n_admissions_skipped_for_auc": pa_n_skipped,
            "classification_report": classification_report(
                y_test_bin, y_pred_bin, digits=3, zero_division=0,
                target_names=["not covered", "covered"], output_dict=True),
            "confusion_matrix": binary_cm.tolist(),
            "recall_at_k": recall_at_k_results,
            "recall_at_k_random_baseline": recall_at_k_random_results,
            "recall_at_k_ceiling": recall_at_k_ceiling_results,
            "recall_at_k_per_admission": per_admission_recall_at_k_results,
            "recall_at_k_per_admission_random_baseline": per_admission_recall_at_k_random_results,
            "recall_at_k_per_admission_ceiling": per_admission_recall_at_k_ceiling_results,
            "precision_at_k": precision_at_k_results,
            "precision_at_k_random_baseline": precision_at_k_random_results,
            "precision_at_k_ceiling": precision_at_k_ceiling_results,
            "precision_at_k_per_admission": per_admission_precision_at_k_results,
            "precision_at_k_per_admission_random_baseline": per_admission_precision_at_k_random_results,
            "precision_at_k_per_admission_ceiling": per_admission_precision_at_k_ceiling_results,
            "recall_positive_class": pos_class,
        },
    }
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        if not isinstance(all_results, list):
            raise ValueError(
                f"{results_path} exists but isn't a JSON array -- "
                f"expected a list of prior run results."
            )
    else:
        all_results = []
    all_results.append(result)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAppended results ({len(all_results)} run(s) total) -> {results_path}")


if __name__ == "__main__":
    main()
