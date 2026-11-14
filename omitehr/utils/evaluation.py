#!/usr/bin/env python3
"""Evaluation utilities and lightweight baselines for sentence ranking.

The labels represent support by the reference summary. By default labels 1 and
2 are positive; --positive-label 2 treats only label 2 as positive. Baselines fit only on training data and
write aggregate metrics without source text or identifiers.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", re.IGNORECASE)
NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
NOTE_DATETIME_RE = re.compile(
    r"NOTE_DATETIME:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Example:
    doc_id: Any
    sentence_idx: int
    text: str
    label: int


def add_label_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--positive-label", "--positive_label", type=int, choices=(1, 2), default=1,
        help="Minimum positive label: 1 includes labels 1 and 2; 2 includes only label 2.",
    )


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens with numeric values collapsed to one token.

    Collapsing numbers keeps dates, IDs, and individual lab values from
    becoming admission-specific vocabulary while retaining the fact that a
    sentence contains a numeric measurement.
    """
    out = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        out.append("<num>" if NUMBER_RE.match(token) else token)
    return out


def parse_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value[:19])
    except ValueError:
        return None


def iter_valid_sentences_with_note_time(row: Mapping[str, Any]):
    """Yield aligned (valid_index, sentence, label, source-note time).

    We walk the unfiltered sentence list so a metadata/header sentence that was
    filtered out can still establish the timestamp for subsequent valid
    sentences from that note.
    """
    valid_sentences = row.get("source_sentences_valid", [])
    # The original discharge-summary pipeline uses ``entailment_labels``;
    # BHC-IM's contextual labeling pipeline stores the same aligned 0/1/2
    # labels under ``contextual_label_values``.
    labels = row.get("contextual_label_values", row.get("entailment_labels", []))
    all_sentences = row.get("source_sentences", [])
    valid_mask = row.get("source_valid_mask", [])
    if len(valid_sentences) != len(labels):
        raise ValueError(
            f"doc_id={row.get('doc_id')} has {len(valid_sentences)} valid "
            f"sentences but {len(labels)} labels"
        )

    if len(all_sentences) != len(valid_mask):
        for index, (sentence, label) in enumerate(zip(valid_sentences, labels)):
            match = NOTE_DATETIME_RE.search(sentence)
            note_time = parse_datetime(match.group(1)) if match else None
            yield index, sentence, label, note_time
        return

    valid_index = 0
    note_time = None
    for sentence, keep in zip(all_sentences, valid_mask):
        match = NOTE_DATETIME_RE.search(sentence)
        if match:
            note_time = parse_datetime(match.group(1))
        if not keep:
            continue
        if valid_index >= len(valid_sentences):
            raise ValueError(f"doc_id={row.get('doc_id')} valid mask has too many true entries")
        # Use the canonical retained text rather than assuming it is byte-for-byte
        # identical to the corresponding pre-filter sentence.
        yield valid_index, valid_sentences[valid_index], labels[valid_index], note_time
        valid_index += 1
    if valid_index != len(valid_sentences):
        raise ValueError(
            f"doc_id={row.get('doc_id')} valid mask selected {valid_index} "
            f"sentences but source_sentences_valid has {len(valid_sentences)}"
        )


def load_examples(
    path: Path,
    cutoff_at_summary: bool = False,
    exclude_verbatim_target: bool = False,
    positive_label: int = 1,
) -> Tuple[List[Example], Dict[str, int]]:
    examples: List[Example] = []
    stats = collections.Counter()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["documents"] += 1
            summary_time = parse_datetime(row.get("note_datetime"))
            target_text = row.get("target_text", "")
            for sentence_idx, sentence, label, note_time in iter_valid_sentences_with_note_time(row):
                if cutoff_at_summary and summary_time and note_time and note_time > summary_time:
                    stats["post_summary_sentences_excluded"] += 1
                    continue
                if exclude_verbatim_target and sentence.strip() and sentence.strip() in target_text:
                    stats["verbatim_target_sentences_excluded"] += 1
                    continue
                if label not in (0, 1, 2):
                    stats["unjudged_sentences"] += 1
                    continue
                binary_label = int(label >= positive_label)
                examples.append(
                    Example(
                        doc_id=row["doc_id"],
                        sentence_idx=sentence_idx,
                        text=sentence,
                        label=binary_label,
                    )
                )
                stats["reference_supported" if binary_label else "reference_absent"] += 1
    stats["judged_sentences"] = len(examples)
    return examples, dict(stats)


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for j in range(start, end):
            ranks[order[j]] = average_rank
        start = end
    return ranks


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Average precision grouped at distinct thresholds (sklearn semantics)."""
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    seen = 0
    previous_recall = 0.0
    ap = 0.0
    start = 0
    while start < len(order):
        score = scores[order[start]]
        end = start
        group_tp = 0
        while end < len(order) and scores[order[end]] == score:
            group_tp += labels[order[end]]
            end += 1
        tp += group_tp
        seen += end - start
        recall = tp / n_pos
        precision = tp / seen
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return ap


def rank_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> Tuple[float, float]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    chosen = order[: min(k, len(order))]
    caught = sum(labels[i] for i in chosen)
    precision = caught / len(chosen) if chosen else 0.0
    total_positive = sum(labels)
    recall = caught / total_positive if total_positive else 0.0
    return precision, recall


def group_by_doc(examples: Sequence[Example], scores: Sequence[float]):
    grouped: Dict[Any, Tuple[List[int], List[float]]] = {}
    for example, score in zip(examples, scores):
        labels, doc_scores = grouped.setdefault(example.doc_id, ([], []))
        labels.append(example.label)
        doc_scores.append(float(score))
    return grouped


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def bootstrap_mean_ci(
    values: Sequence[float], rng: random.Random, n_bootstrap: int,
) -> List[float] | None:
    if not values or n_bootstrap <= 0:
        return None
    boot = []
    for _ in range(n_bootstrap):
        boot.append(sum(rng.choice(values) for _ in values) / len(values))
    return [percentile(boot, 0.025), percentile(boot, 0.975)]  # type: ignore[list-item]


def evaluate(
    examples: Sequence[Example],
    scores: Sequence[float],
    k_values: Sequence[int],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    labels = [example.label for example in examples]
    grouped = group_by_doc(examples, scores)
    per_doc: Dict[str, List[float]] = collections.defaultdict(list)

    for doc_labels, doc_scores in grouped.values():
        auc = roc_auc(doc_labels, doc_scores)
        ap = average_precision(doc_labels, doc_scores)
        if auc is not None:
            per_doc["auroc"].append(auc)
        if ap is not None:
            per_doc["auprc"].append(ap)
        for k in k_values:
            precision, recall = rank_at_k(doc_labels, doc_scores, k)
            per_doc[f"precision_at_{k}"].append(precision)
            if sum(doc_labels):
                per_doc[f"recall_at_{k}"].append(recall)

    rng = random.Random(seed)
    macro = {}
    for metric, values in sorted(per_doc.items()):
        macro[metric] = {
            "mean": mean(values),
            "bootstrap_95_ci": bootstrap_mean_ci(values, rng, bootstrap),
            "n_admissions": len(values),
        }

    return {
        "n_sentences": len(examples),
        "n_admissions": len(grouped),
        "positive_prevalence": sum(labels) / len(labels),
        "pooled": {
            "auroc": roc_auc(labels, scores),
            "auprc": average_precision(labels, scores),
        },
        "admission_macro": macro,
    }


class TokenNaiveBayes:
    def __init__(self, alpha: float = 1.0, min_df: int = 3):
        self.alpha = alpha
        self.min_df = min_df
        self.log_ratio: Dict[str, float] = {}
        self.unknown_log_ratio = 0.0
        self.prior_log_odds = 0.0
        self.vocabulary_size = 0

    def fit(self, examples: Iterable[Example]) -> None:
        token_counts = [collections.Counter(), collections.Counter()]
        document_frequency = collections.Counter()
        total_tokens = [0, 0]
        class_examples = [0, 0]

        for example in examples:
            tokens = tokenize(example.text)
            class_examples[example.label] += 1
            token_counts[example.label].update(tokens)
            total_tokens[example.label] += len(tokens)
            document_frequency.update(set(tokens))

        vocabulary = {token for token, df in document_frequency.items() if df >= self.min_df}
        self.vocabulary_size = len(vocabulary)
        denominators = [
            total_tokens[label] + self.alpha * (self.vocabulary_size + 1)
            for label in (0, 1)
        ]
        self.unknown_log_ratio = math.log(self.alpha / denominators[1]) - math.log(
            self.alpha / denominators[0]
        )
        self.log_ratio = {
            token: math.log((token_counts[1][token] + self.alpha) / denominators[1])
            - math.log((token_counts[0][token] + self.alpha) / denominators[0])
            for token in vocabulary
        }
        self.prior_log_odds = math.log(
            (class_examples[1] + self.alpha) / (class_examples[0] + self.alpha)
        )

    def score(self, text: str, normalize: bool = False) -> float:
        tokens = tokenize(text)
        evidence = sum(self.log_ratio.get(token, self.unknown_log_ratio) for token in tokens)
        if normalize:
            evidence /= max(1, len(tokens))
        return self.prior_log_odds + evidence


def deterministic_random_score(example: Example, seed: int) -> float:
    value = f"{seed}|{example.doc_id}|{example.sentence_idx}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") / 2**64


def repetition_scores(examples: Sequence[Example]) -> List[float]:
    counts: Dict[Any, collections.Counter] = collections.defaultdict(collections.Counter)
    normalized = []
    for example in examples:
        key = " ".join(tokenize(example.text))
        normalized.append(key)
        counts[example.doc_id][key] += 1
    return [float(counts[example.doc_id][key]) for example, key in zip(examples, normalized)]


def choose_direction(examples: Sequence[Example], raw_scores: Sequence[float]) -> int:
    labels = [example.label for example in examples]
    auc = roc_auc(labels, raw_scores)
    return 1 if auc is None or auc >= 0.5 else -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_label_argument(parser)
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--eval_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_predictions_dir", type=Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min_df", type=int, default=3)
    parser.add_argument("--k", type=str, default="1,5,10")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cutoff_at_summary",
        action="store_true",
        help="Exclude candidate sentences from notes timestamped after note_datetime.",
    )
    parser.add_argument(
        "--exclude_verbatim_target",
        action="store_true",
        help=(
            "Diagnostic only: exclude source sentences copied verbatim into the discharge summary. "
            "This uses the reference and is not an inference-time feature."
        ),
    )
    args = parser.parse_args()

    k_values = [int(value) for value in args.k.split(",") if value.strip()]
    train_examples, train_stats = load_examples(
        args.train_jsonl,
        cutoff_at_summary=args.cutoff_at_summary,
        exclude_verbatim_target=args.exclude_verbatim_target,
        positive_label=args.positive_label,
    )
    eval_examples, eval_stats = load_examples(
        args.eval_jsonl,
        cutoff_at_summary=args.cutoff_at_summary,
        exclude_verbatim_target=args.exclude_verbatim_target,
        positive_label=args.positive_label,
    )
    if not train_examples or not eval_examples:
        raise ValueError("Both train and evaluation files need at least one judged sentence")

    print(f"Loaded {len(train_examples):,} judged train sentences from {train_stats['documents']} admissions")
    print(f"Loaded {len(eval_examples):,} judged eval sentences from {eval_stats['documents']} admissions")

    train_position = [example.sentence_idx for example in train_examples]
    train_length = [len(tokenize(example.text)) for example in train_examples]
    position_direction = choose_direction(train_examples, train_position)
    length_direction = choose_direction(train_examples, train_length)

    nb = TokenNaiveBayes(alpha=args.alpha, min_df=args.min_df)
    nb.fit(train_examples)
    print(f"Fitted token Naive Bayes with {nb.vocabulary_size:,} tokens")

    scorers: Mapping[str, Callable[[Example], float]] = {
        "random": lambda example: deterministic_random_score(example, args.seed),
        "position": lambda example: position_direction * example.sentence_idx,
        "length": lambda example: length_direction * len(tokenize(example.text)),
        "token_nb": lambda example: nb.score(example.text, normalize=False),
        "token_nb_mean": lambda example: nb.score(example.text, normalize=True),
    }

    results = {}
    def metric_text(value):
        return "NA" if value is None else f"{value:.4f}"

    for model_name, scorer in scorers.items():
        scores = [scorer(example) for example in eval_examples]
        if args.output_predictions_dir:
            args.output_predictions_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_predictions_dir / f"{model_name}.jsonl").open("w") as f:
                for ex, score in zip(eval_examples, scores):
                    f.write(json.dumps({"doc_id": str(ex.doc_id), "sentence_idx": int(ex.sentence_idx), "label": int(ex.label), "score": float(score)}) + "\n")
        results[model_name] = evaluate(
            eval_examples, scores, k_values, args.bootstrap, args.seed
        )
        pooled = results[model_name]["pooled"]
        print(
            f"{model_name:14s} AUROC={metric_text(pooled['auroc'])} "
            f"AUPRC={metric_text(pooled['auprc'])}"
        )

    repetition = repetition_scores(eval_examples)
    if args.output_predictions_dir:
        args.output_predictions_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_predictions_dir / "repetition.jsonl").open("w") as f:
            for ex, score in zip(eval_examples, repetition):
                f.write(json.dumps({"doc_id": str(ex.doc_id), "sentence_idx": int(ex.sentence_idx), "label": int(ex.label), "score": float(score)}) + "\n")
    results["repetition"] = evaluate(
        eval_examples, repetition, k_values, args.bootstrap, args.seed
    )
    pooled = results["repetition"]["pooled"]
    print(f"{'repetition':14s} AUROC={metric_text(pooled['auroc'])} AUPRC={metric_text(pooled['auprc'])}")

    output = {
        "task": "predict complete discharge-summary reference support for each EHR sentence",
        "positive_label": args.positive_label,
        "warning": (
            "Reference support is weak supervision for clinical relevance; "
            "it is not a clinician-validated should-include label."
        ),
        # Keep released metrics portable and free of local filesystem usernames.
        "train_jsonl": args.train_jsonl.name,
        "eval_jsonl": args.eval_jsonl.name,
        "config": {
            "alpha": args.alpha,
            "min_df": args.min_df,
            "k_values": k_values,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "position_direction": position_direction,
            "length_direction": length_direction,
            "numeric_token_normalization": True,
            "cutoff_at_summary": args.cutoff_at_summary,
            "exclude_verbatim_target": args.exclude_verbatim_target,
        },
        "train_stats": train_stats,
        "eval_stats": eval_stats,
        "models": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved aggregate results -> {args.output_json}")


if __name__ == "__main__":
    main()
