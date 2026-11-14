#!/usr/bin/env python3
"""Score sentence embeddings jointly within an admission.

The model combines frozen sentence embeddings with note and section metadata,
then applies a small transformer over each admission.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from omitehr.utils.metadata import SECTION_NAMES, extract_document_metadata
from omitehr.utils.evaluation import Example, evaluate, add_label_argument


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Record:
    doc_id: str
    embeddings: np.ndarray
    numeric: np.ndarray
    title_ids: np.ndarray
    section_ids: np.ndarray
    labels: np.ndarray
    sentence_indices: np.ndarray
    texts: tuple[str, ...] = ()


def _load_json_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open() as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def load_records(
    records_jsonl: Path,
    embeddings_npz: Path,
    max_sentences: int = 2048,
    chunked_blocks: bool = False, positive_label: int = 1,
) -> tuple[list[Record], Dict[str, Any]]:
    """Join frozen embeddings to metadata and group rows by admission."""
    if chunked_blocks and max_sentences <= 0:
        raise ValueError("--max_sentences must be positive with --chunked_blocks")
    data = np.load(embeddings_npz, allow_pickle=False)
    required = {"embeddings", "labels", "doc_ids", "sentence_idx"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{embeddings_npz} is missing arrays: {sorted(missing)}")
    vectors = data["embeddings"]
    labels = data["labels"]
    keys = [(str(doc_id), int(sentence_idx)) for doc_id, sentence_idx in zip(data["doc_ids"], data["sentence_idx"])]
    if len(vectors) != len(labels) or len(vectors) != len(keys):
        raise ValueError(f"Embedding arrays are unaligned in {embeddings_npz}")
    vector_by_key = {key: (vectors[index], labels[index]) for index, key in enumerate(keys)}

    output: list[Record] = []
    n_unlabeled = 0
    n_truncated = 0
    for row in _load_json_rows(records_jsonl):
        doc_id = str(row["doc_id"])
        metadata = extract_document_metadata(row, local_window=0)
        selected: list[tuple[int, np.ndarray, float]] = []
        for sentence_idx in range(len(metadata.sentence_text)):
            item = vector_by_key.get((doc_id, sentence_idx))
            if item is None:
                continue
            vector, label = item
            if not np.isfinite(label):
                n_unlabeled += 1
                continue
            selected.append((sentence_idx, vector, label))
        if not selected:
            continue

        if chunked_blocks:
            n_blocks = math.ceil(len(selected) / max_sentences)
            blocks = np.array_split(np.arange(len(selected)), n_blocks)
        else:
            # Preserve the whole record when possible. For unusually long
            # records, retain evenly spaced sentences so early, middle, and
            # late notes remain represented rather than silently taking only
            # the beginning.
            if max_sentences > 0 and len(selected) > max_sentences:
                chosen = np.linspace(0, len(selected) - 1, max_sentences, dtype=np.int64)
                selected = [selected[int(index)] for index in chosen]
                n_truncated += 1
            blocks = [np.arange(len(selected), dtype=np.int64)]
        for block in blocks:
            block_selected = [selected[int(index)] for index in block]
            sentence_indices = np.asarray(
                [item[0] for item in block_selected], dtype=np.int64
            )
            output.append(
                Record(
                    doc_id=doc_id,
                    embeddings=np.asarray(
                        [item[1] for item in block_selected], dtype=np.float32
                    ),
                    numeric=metadata.numeric[sentence_indices],
                    title_ids=metadata.note_title_bucket[sentence_indices],
                    section_ids=metadata.section_id[sentence_indices],
                    labels=np.asarray(
                        [int(item[2] >= positive_label) for item in block_selected], dtype=np.float32
                    ),
                    sentence_indices=sentence_indices,
                    texts=tuple(
                        metadata.sentence_text[int(i)] for i in sentence_indices
                    ),
                )
            )
    if not output:
        raise ValueError(f"No labeled admissions could be joined from {records_jsonl} and {embeddings_npz}")
    return output, {
        "records_jsonl": str(records_jsonl.resolve()),
        "embeddings_npz": str(embeddings_npz.resolve()),
        "n_admissions": len(output),
        "n_sentences": int(sum(len(record.labels) for record in output)),
        "n_positive": int(sum(record.labels.sum() for record in output)),
        "n_unlabeled_dropped": n_unlabeled,
        "n_records_truncated": n_truncated,
        "max_sentences": max_sentences,
        "chunked_blocks": chunked_blocks,
        "n_blocks": len(output),
    }


class RecordDataset(Dataset):
    def __init__(self, records: Sequence[Record]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Record:
        return self.records[index]


def collate_records(records: Sequence[Record]) -> Dict[str, Any]:
    max_length = max(len(record.labels) for record in records)
    embedding_dim = records[0].embeddings.shape[1]
    batch_size = len(records)
    embeddings = np.zeros((batch_size, max_length, embedding_dim), dtype=np.float32)
    numeric = np.zeros((batch_size, max_length, 7), dtype=np.float32)
    title_ids = np.zeros((batch_size, max_length), dtype=np.int64)
    section_ids = np.zeros((batch_size, max_length), dtype=np.int64)
    labels = np.zeros((batch_size, max_length), dtype=np.float32)
    mask = np.zeros((batch_size, max_length), dtype=bool)
    doc_ids: list[str] = []
    sentence_indices: list[np.ndarray] = []
    for row, record in enumerate(records):
        length = len(record.labels)
        embeddings[row, :length] = record.embeddings
        numeric[row, :length] = record.numeric
        title_ids[row, :length] = record.title_ids
        section_ids[row, :length] = record.section_ids
        labels[row, :length] = record.labels
        mask[row, :length] = True
        doc_ids.append(record.doc_id)
        sentence_indices.append(record.sentence_indices)
    return {
        "embeddings": torch.from_numpy(embeddings),
        "numeric": torch.from_numpy(numeric),
        "title_ids": torch.from_numpy(title_ids),
        "section_ids": torch.from_numpy(section_ids),
        "labels": torch.from_numpy(labels),
        "mask": torch.from_numpy(mask),
        "doc_ids": doc_ids,
        "sentence_indices": sentence_indices,
    }


class RecordContextTransformer(nn.Module):
    """Small admission transformer over frozen sentence representations."""

    def __init__(
        self,
        embedding_dim: int,
        context_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        max_positions: int = 2048,
        title_buckets: int = 128,
    ):
        super().__init__()
        if context_dim % heads:
            raise ValueError("context_dim must be divisible by heads")
        self.layers = layers
        self.embedding_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, context_dim)
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(7, 32), nn.GELU(), nn.LayerNorm(32)
        )
        self.title_embedding = nn.Embedding(title_buckets, 16)
        self.section_embedding = nn.Embedding(len(SECTION_NAMES), 8)
        self.structural_projection = nn.Sequential(
            nn.Linear(56, context_dim), nn.GELU(), nn.LayerNorm(context_dim)
        )
        self.position_embedding = nn.Embedding(max_positions, context_dim)
        # ``layers=0`` is the matched no-context ablation: it retains the
        # exact sentence/metadata representation and scoring head while
        # removing every cross-sentence operation.
        if layers < 0:
            raise ValueError("layers must be non-negative")
        if layers == 0:
            self.context_encoder = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=context_dim,
                nhead=heads,
                dim_feedforward=4 * context_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Dropout(dropout), nn.Linear(context_dim, 1)
        )
        self.max_positions = max_positions

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        embeddings = self.embedding_projection(batch["embeddings"])
        numeric = self.numeric_projection(batch["numeric"])
        title = self.title_embedding(batch["title_ids"])
        section = self.section_embedding(batch["section_ids"])
        structural = torch.cat((numeric, title, section), dim=-1)
        structural = self.structural_projection(structural)
        positions = torch.arange(
            embeddings.shape[1], device=embeddings.device
        ).clamp_max(self.max_positions - 1)
        hidden = embeddings + structural + self.position_embedding(positions)[None, :, :]
        if self.layers:
            hidden = self.context_encoder(hidden, src_key_padding_mask=~batch["mask"])
        return self.output(hidden).squeeze(-1)


def listwise_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Encourage probability mass on all positive sentences in each record."""
    losses = []
    for row in range(logits.shape[0]):
        valid = mask[row]
        positives = valid & (labels[row] >= 0.5)
        if not positives.any() or (~positives & valid).sum() == 0:
            continue
        losses.append(
            torch.logsumexp(logits[row, valid], dim=0)
            - torch.logsumexp(logits[row, positives], dim=0)
        )
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    max_examples: int = 32,
) -> torch.Tensor:
    """Hard-positive/hard-negative ranking loss within each admission.

    Only the lowest-scoring positives and highest-scoring negatives are kept,
    which focuses training on the decisions that affect a top-k review budget
    without materializing every positive/negative pair in long records.
    """
    losses = []
    for row in range(logits.shape[0]):
        valid = mask[row]
        positives = logits[row, valid & (labels[row] >= 0.5)]
        negatives = logits[row, valid & (labels[row] < 0.5)]
        if not len(positives) or not len(negatives):
            continue
        positives = torch.topk(positives, min(max_examples, len(positives)), largest=False).values
        negatives = torch.topk(negatives, min(max_examples, len(negatives)), largest=True).values
        losses.append(torch.nn.functional.softplus(negatives[:, None] - positives[None, :]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_logits = logits[mask]
    valid_labels = labels[mask]
    positive = valid_labels.sum().clamp_min(1.0)
    negative = (valid_labels.numel() - positive).clamp_min(1.0)
    return nn.functional.binary_cross_entropy_with_logits(
        valid_logits, valid_labels, pos_weight=negative / positive
    )


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[Example], list[float]]:
    model.eval()
    examples: list[Example] = []
    scores: list[float] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        probabilities = torch.sigmoid(model(batch)).cpu().numpy()
        for row, doc_id in enumerate(raw_batch["doc_ids"]):
            length = len(raw_batch["sentence_indices"][row])
            for col in range(length):
                examples.append(Example(doc_id, int(raw_batch["sentence_indices"][row][col]), "", 0))
                scores.append(float(probabilities[row, col]))
    return examples, scores


def attach_labels(examples: Sequence[Example], records: Sequence[Record]) -> list[Example]:
    labels = {
        (record.doc_id, int(sentence_idx)): int(label)
        for record in records
        for sentence_idx, label in zip(record.sentence_indices, record.labels)
    }
    return [Example(example.doc_id, example.sentence_idx, "", labels[(example.doc_id, example.sentence_idx)]) for example in examples]


@dataclass
class Config:
    embedding_dim: int
    context_dim: int
    layers: int
    heads: int
    dropout: float
    max_positions: int
    title_buckets: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    ranking_weight: float
    pairwise_weight: float
    patience: int
    seed: int
    positive_label: int = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_label_argument(parser)
    parser.add_argument("--train_records_jsonl", type=Path, required=True)
    parser.add_argument("--dev_records_jsonl", type=Path, required=True)
    parser.add_argument("--test_records_jsonl", type=Path, required=True)
    parser.add_argument("--train_embeddings_npz", type=Path, required=True)
    parser.add_argument("--dev_embeddings_npz", type=Path, required=True)
    parser.add_argument("--test_embeddings_npz", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test_predictions", type=Path)
    parser.add_argument("--max_sentences", type=int, default=2048)
    parser.add_argument(
        "--chunked_blocks",
        action="store_true",
        help="Retain all labeled sentences and score them in contiguous balanced blocks.",
    )
    parser.add_argument("--context_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--ranking_weight", type=float, default=1.0)
    parser.add_argument("--pairwise_weight", type=float, default=0.0)
    parser.add_argument("--pairwise_examples", type=int, default=32)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--k", default="5,10,20,50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train, train_stats = load_records(
        args.train_records_jsonl,
        args.train_embeddings_npz,
        args.max_sentences,
        args.chunked_blocks, positive_label=args.positive_label,
    )
    dev, dev_stats = load_records(
        args.dev_records_jsonl,
        args.dev_embeddings_npz,
        args.max_sentences,
        args.chunked_blocks, positive_label=args.positive_label,
    )
    test, test_stats = load_records(
        args.test_records_jsonl,
        args.test_embeddings_npz,
        args.max_sentences,
        args.chunked_blocks, positive_label=args.positive_label,
    )
    embedding_dim = int(train[0].embeddings.shape[1])
    config = Config(
        embedding_dim=embedding_dim, context_dim=args.context_dim, layers=args.layers,
        heads=args.heads, dropout=args.dropout, max_positions=args.max_sentences,
        title_buckets=128, batch_size=args.batch_size, epochs=args.epochs,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        ranking_weight=args.ranking_weight, patience=args.patience, seed=args.seed,
        pairwise_weight=args.pairwise_weight, positive_label=args.positive_label,
    )
    train_loader = DataLoader(RecordDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate_records)
    dev_loader = DataLoader(RecordDataset(dev), batch_size=args.batch_size, shuffle=False, collate_fn=collate_records)
    test_loader = DataLoader(RecordDataset(test), batch_size=args.batch_size, shuffle=False, collate_fn=collate_records)
    model = RecordContextTransformer(
        embedding_dim, args.context_dim, args.layers, args.heads, args.dropout,
        args.max_sentences, 128,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    k_values = [int(value) for value in args.k.split(",") if value.strip()]
    best_score = -math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch)
                bce = masked_bce(logits, batch["labels"], batch["mask"])
                ranking = listwise_loss(logits, batch["labels"], batch["mask"])
                pairwise = pairwise_ranking_loss(
                    logits, batch["labels"], batch["mask"], args.pairwise_examples
                )
                loss = bce + args.ranking_weight * ranking + args.pairwise_weight * pairwise
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
        dev_examples, dev_scores = predict(model, dev_loader, device)
        dev_examples = attach_labels(dev_examples, dev)
        dev_metrics = evaluate(dev_examples, dev_scores, k_values, bootstrap=0, seed=args.seed)
        score = float(dev_metrics["admission_macro"]["auprc"]["mean"])
        history.append({"epoch": epoch, "loss": total_loss / max(1, len(train_loader)), "dev_macro_auprc": score})
        print(f"epoch={epoch:03d} loss={total_loss / max(1, len(train_loader)):.5f} dev_macro_AUPRC={score:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_examples, test_scores = predict(model, test_loader, device)
    test_examples = attach_labels(test_examples, test)
    output = {
        "task": (
            "context-free matched-metadata reference-inclusion ranking"
            if args.layers == 0 else "admission-contextual reference-inclusion ranking"
        ),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_block_count": len(train),
        "dev_block_count": len(dev),
        "test_block_count": len(test),
        "train_admission_count": len({record.doc_id for record in train}),
        "dev_admission_count": len({record.doc_id for record in dev}),
        "test_admission_count": len({record.doc_id for record in test}),
        "config": asdict(config), "train": train_stats, "dev": dev_stats, "test": test_stats,
        "best_epoch": best_epoch, "history": history,
        "test_metrics": evaluate(test_examples, test_scores, k_values, args.bootstrap, args.seed),
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, args.checkpoint)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    if args.test_predictions:
        args.test_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.test_predictions.open("w") as output_file:
            for example, score in zip(test_examples, test_scores):
                output_file.write(json.dumps({"doc_id": example.doc_id, "sentence_idx": example.sentence_idx, "label": example.label, "score": score}) + "\n")
    print(f"Saved checkpoint -> {args.checkpoint}")
    print(f"Saved results -> {args.output_json}")


if __name__ == "__main__":
    main()
