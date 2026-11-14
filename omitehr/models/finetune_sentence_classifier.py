#!/usr/bin/env python3
"""Fine-tune a clinical encoder to score source sentences.

Training uses sentence-level labels and optional admission metadata. The
reference summary supplies labels only; it is never model input.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from omitehr.utils.metadata import (
    SECTION_NAMES,
    extract_document_metadata,
    METADATA_GROUPS,
    NUMERIC_GROUP_INDICES,
    parse_metadata_groups,
)
from omitehr.utils.evaluation import Example, evaluate, add_label_argument


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encoder_layers(encoder: nn.Module) -> Sequence[nn.Module]:
    """Return transformer blocks from common Hugging Face encoder layouts."""
    bert_encoder = getattr(encoder, "encoder", None)
    layers = getattr(bert_encoder, "layer", None)
    if layers is None:
        layers = getattr(encoder, "layers", None)
    if layers is None:
        raise ValueError(
            "Encoder does not expose transformer blocks at encoder.layer or layers; "
            "use --unfreeze_last_n_layers -1 to fine-tune all parameters."
        )
    return layers


def encoder_accepts_token_type_ids(encoder: nn.Module) -> bool:
    """Whether a model's forward API accepts BERT segment IDs."""
    return "token_type_ids" in inspect.signature(encoder.forward).parameters


@dataclass(frozen=True)
class SentenceRecord:
    doc_id: str
    sentence_idx: int
    text: str
    label: int
    numeric_metadata: np.ndarray
    note_title_bucket: int
    section_id: int


def load_records(
    path: Path, local_window: int = 0, title_buckets: int = 128, positive_label: int = 1,
) -> tuple[list[SentenceRecord], Dict[str, Any]]:
    records: list[SentenceRecord] = []
    n_unjudged = 0
    doc_ids: set[str] = set()
    with path.open() as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = extract_document_metadata(
                row, local_window=local_window, title_buckets=title_buckets
            )
            sentences = row["source_sentences_valid"]
            labels = row.get("contextual_label_values", row.get("entailment_labels", []))
            if len(sentences) != len(labels) or len(sentences) != len(metadata.numeric):
                raise ValueError(f"Unaligned sentences/labels/metadata for doc_id={row['doc_id']}")
            doc_id = str(row["doc_id"])
            doc_ids.add(doc_id)
            for sentence_idx, (text, label) in enumerate(zip(sentences, labels)):
                if label is None:
                    n_unjudged += 1
                    continue
                records.append(
                    SentenceRecord(
                        doc_id=doc_id,
                        sentence_idx=sentence_idx,
                        text=text,
                        label=int(label >= positive_label),
                        numeric_metadata=metadata.numeric[sentence_idx],
                        note_title_bucket=int(metadata.note_title_bucket[sentence_idx]),
                        section_id=int(metadata.section_id[sentence_idx]),
                    )
                )
    return records, {
        "path": str(path.resolve()),
        "n_admissions": len(doc_ids),
        "n_sentences": len(records),
        "n_positive": sum(record.label for record in records),
        "n_unjudged_filtered": n_unjudged,
    }


class SentenceDataset(Dataset):
    def __init__(self, records: Sequence[SentenceRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SentenceRecord:
        return self.records[index]


class AdmissionChunkBatchSampler(Sampler[list[int]]):
    """Visit every sentence once per epoch while retaining within-admission pairs."""

    def __init__(
        self,
        records: Sequence[SentenceRecord],
        sentences_per_admission: int,
        admissions_per_batch: int,
        seed: int,
    ):
        self.sentences_per_admission = sentences_per_admission
        self.admissions_per_batch = admissions_per_batch
        self.seed = seed
        self.epoch = 0
        by_doc: Dict[str, list[int]] = {}
        for index, record in enumerate(records):
            by_doc.setdefault(record.doc_id, []).append(index)
        self.by_doc = by_doc
        self.n_chunks = sum(
            math.ceil(len(indices) / sentences_per_admission)
            for indices in self.by_doc.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(self.n_chunks / self.admissions_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        chunks: list[list[int]] = []
        for indices in self.by_doc.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            chunks.extend(
                shuffled[start:start + self.sentences_per_admission]
                for start in range(0, len(shuffled), self.sentences_per_admission)
            )
        rng.shuffle(chunks)
        for start in range(0, len(chunks), self.admissions_per_batch):
            yield [index for chunk in chunks[start:start + self.admissions_per_batch] for index in chunk]


class TokenizingCollator:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, records: Sequence[SentenceRecord]) -> Dict[str, Any]:
        tokens = self.tokenizer(
            [record.text for record in records],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            **tokens,
            "labels": torch.tensor([record.label for record in records], dtype=torch.float32),
            "numeric_metadata": torch.from_numpy(
                np.stack([record.numeric_metadata for record in records])
            ),
            "note_title_bucket": torch.tensor(
                [record.note_title_bucket for record in records], dtype=torch.long
            ),
            "section_id": torch.tensor([record.section_id for record in records], dtype=torch.long),
            "position_ids": torch.tensor([record.sentence_idx for record in records], dtype=torch.long),
            "doc_ids": [record.doc_id for record in records],
            "sentence_idx": [record.sentence_idx for record in records],
        }


class FineTunedClinicalBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        metadata_groups: Sequence[str],
        title_buckets: int = 128,
        dropout: float = 0.1,
        unfreeze_last_n_layers: int = 4,
        representation: str = "legacy",
        context_dim: int = 256,
        position_capacity: int = 1024,
    ):
        super().__init__()
        unknown = set(metadata_groups).difference(METADATA_GROUPS)
        if unknown:
            raise ValueError(f"Unknown metadata groups: {sorted(unknown)}")
        self.metadata_groups = tuple(metadata_groups)
        if representation not in {"legacy", "context_transformer"}:
            raise ValueError(f"Unknown representation: {representation}")
        self.representation = representation
        self.encoder = AutoModel.from_pretrained(model_name)
        # See omitehr.utils.embeddings: some environments have Triton importable
        # but cannot build its optional compiled ModernBERT path, so disable it.
        if getattr(self.encoder.config, "model_type", None) == "modernbert":
            self.encoder.config.reference_compile = False
        self._configure_encoder_trainability(unfreeze_last_n_layers)
        hidden_dim = int(self.encoder.config.hidden_size)
        numeric_mask = torch.zeros(7, dtype=torch.float32)
        for group in self.metadata_groups:
            for index in NUMERIC_GROUP_INDICES.get(group, ()):
                numeric_mask[index] = 1.0
        self.register_buffer("numeric_metadata_mask", numeric_mask, persistent=False)
        numeric_dim = 32 if representation == "context_transformer" else 16
        self.numeric_projection = nn.Sequential(
            nn.Linear(7, numeric_dim), nn.GELU(), nn.LayerNorm(numeric_dim)
        )
        self.title_embedding = nn.Embedding(title_buckets, 16)
        self.section_embedding = nn.Embedding(len(SECTION_NAMES), 8)
        if representation == "context_transformer":
            # Match RecordContextTransformer exactly up to (but excluding)
            # its cross-sentence TransformerEncoder.
            self.embedding_projection = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, context_dim)
            )
            self.structural_projection = nn.Sequential(
                nn.Linear(numeric_dim + 16 + 8, context_dim),
                nn.GELU(),
                nn.LayerNorm(context_dim),
            )
            self.position_embedding = nn.Embedding(position_capacity, context_dim)
            self.position_capacity = position_capacity
            self.classifier = nn.Sequential(
                nn.LayerNorm(context_dim), nn.Dropout(dropout), nn.Linear(context_dim, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim + numeric_dim + 16 + 8, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

    def _configure_encoder_trainability(self, unfreeze_last_n_layers: int) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = unfreeze_last_n_layers < 0
        if unfreeze_last_n_layers == 0 or unfreeze_last_n_layers < 0:
            return
        layers = encoder_layers(self.encoder)
        if unfreeze_last_n_layers > len(layers):
            raise ValueError(
                f"Requested {unfreeze_last_n_layers} layers, encoder has {len(layers)}"
            )
        for layer in layers[-unfreeze_last_n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        encoder_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if "token_type_ids" in batch and encoder_accepts_token_type_ids(self.encoder):
            encoder_inputs["token_type_ids"] = batch["token_type_ids"]
        encoded = self.encoder(**encoder_inputs).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).type_as(encoded)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)
        metadata = torch.cat(
            (
                self.numeric_projection(
                    batch["numeric_metadata"] * self.numeric_metadata_mask
                ),
                self.title_embedding(batch["note_title_bucket"])
                * float("title" in self.metadata_groups),
                self.section_embedding(batch["section_id"])
                * float("section" in self.metadata_groups),
            ),
            dim=-1,
        )
        if self.representation == "context_transformer":
            return self.classifier(
                self.embedding_projection(pooled)
                + self.structural_projection(metadata)
                + self.position_embedding(
                    batch["position_ids"].clamp_max(self.position_capacity - 1)
                )
            ).squeeze(-1)
        return self.classifier(torch.cat((pooled, metadata), dim=-1)).squeeze(-1)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def flat_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    doc_ids: Sequence[str],
    max_examples: int,
) -> torch.Tensor:
    losses = []
    for doc_id in dict.fromkeys(doc_ids):
        indices = torch.tensor(
            [index for index, value in enumerate(doc_ids) if value == doc_id],
            device=logits.device,
        )
        doc_logits = logits[indices]
        doc_labels = labels[indices]
        positives = doc_logits[doc_labels >= 0.5]
        negatives = doc_logits[doc_labels < 0.5]
        if not len(positives) or not len(negatives):
            continue
        positives = torch.topk(positives, min(max_examples, len(positives)), largest=False).values
        negatives = torch.topk(negatives, min(max_examples, len(negatives)), largest=True).values
        losses.append(torch.nn.functional.softplus(negatives[:, None] - positives[None, :]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


@torch.no_grad()
def predict(
    model: FineTunedClinicalBERTClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[Example], list[float]]:
    model.eval()
    examples: list[Example] = []
    scores: list[float] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        probabilities = torch.sigmoid(model(batch)).cpu().tolist()
        for index, score in enumerate(probabilities):
            examples.append(
                Example(
                    doc_id=raw_batch["doc_ids"][index],
                    sentence_idx=raw_batch["sentence_idx"][index],
                    text="",
                    label=int(raw_batch["labels"][index]),
                )
            )
            scores.append(float(score))
    return examples, scores


def write_predictions(
    path: Path, examples: Sequence[Example], scores: Sequence[float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        for example, score in zip(examples, scores):
            output_file.write(
                json.dumps(
                    {
                        "doc_id": str(example.doc_id),
                        "sentence_idx": int(example.sentence_idx),
                        "label": int(example.label),
                        "score": float(score),
                    }
                )
                + "\n"
            )


@dataclass
class Config:
    model_name: str
    max_length: int
    epochs: int
    sentences_per_admission: int
    admissions_per_batch: int
    eval_batch_size: int
    encoder_learning_rate: float
    head_learning_rate: float
    weight_decay: float
    warmup_ratio: float
    dropout: float
    unfreeze_last_n_layers: int
    ranking_weight: float
    ranking_examples: int
    title_buckets: int
    metadata_groups: list[str]
    representation: str
    context_dim: int
    position_capacity: int
    patience: int
    seed: int
    positive_label: int = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_label_argument(parser)
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--dev_jsonl", type=Path, required=True)
    parser.add_argument("--test_jsonl", type=Path)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test_predictions",
        type=Path,
        help="Optional per-sentence JSONL for the held-out test split",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Path or model identifier for the encoder to fine-tune.",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--sentences_per_admission", type=int, default=8)
    parser.add_argument("--admissions_per_batch", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--encoder_learning_rate", type=float, default=2e-5)
    parser.add_argument("--head_learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=4)
    parser.add_argument("--ranking_weight", type=float, default=0.2)
    parser.add_argument("--ranking_examples", type=int, default=8)
    parser.add_argument("--title_buckets", type=int, default=128)
    parser.add_argument("--metadata_groups", type=parse_metadata_groups, default=list(METADATA_GROUPS))
    parser.add_argument(
        "--representation", choices=("legacy", "context_transformer"), default="legacy",
        help="Use the contextual transformer's pre-attention representation.",
    )
    parser.add_argument("--context_dim", type=int, default=256)
    parser.add_argument("--position_capacity", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--k", default="5,10,20,50")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = Config(
        **{
            key: getattr(args, key)
            for key in Config.__dataclass_fields__
        }
    )
    set_seed(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train, train_stats = load_records(args.train_jsonl, title_buckets=config.title_buckets, positive_label=config.positive_label)
    dev, dev_stats = load_records(args.dev_jsonl, title_buckets=config.title_buckets, positive_label=config.positive_label)
    test, test_stats = ([], None)
    if args.test_jsonl:
        test, test_stats = load_records(args.test_jsonl, title_buckets=config.title_buckets, positive_label=config.positive_label)
    split_ids = {
        "train": {record.doc_id for record in train},
        "dev": {record.doc_id for record in dev},
        "test": {record.doc_id for record in test},
    }
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise ValueError(f"{left}/{right} leakage: {len(overlap)} shared admissions")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    collator = TokenizingCollator(tokenizer, config.max_length)
    train_sampler = AdmissionChunkBatchSampler(
        train,
        config.sentences_per_admission,
        config.admissions_per_batch,
        config.seed,
    )
    train_loader = DataLoader(
        SentenceDataset(train), batch_sampler=train_sampler, collate_fn=collator
    )
    dev_loader = DataLoader(
        SentenceDataset(dev), batch_size=config.eval_batch_size, collate_fn=collator
    )
    test_loader = (
        DataLoader(SentenceDataset(test), batch_size=config.eval_batch_size, collate_fn=collator)
        if test else None
    )
    model = FineTunedClinicalBERTClassifier(
        config.model_name,
        config.metadata_groups,
        config.title_buckets,
        config.dropout,
        config.unfreeze_last_n_layers,
        config.representation,
        config.context_dim,
        config.position_capacity,
    ).to(device)
    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": config.encoder_learning_rate},
            {"params": head_parameters, "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )
    n_positive = sum(record.label for record in train)
    positive_weight = torch.tensor((len(train) - n_positive) / n_positive, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction="mean")
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    k_values = [int(value) for value in args.k.split(",") if value.strip()]
    best_score = -math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        n_batches = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch)
                classification_loss = bce(logits, batch["labels"])
                ranking_loss = flat_ranking_loss(
                    logits,
                    batch["labels"],
                    raw_batch["doc_ids"],
                    config.ranking_examples,
                )
                loss = classification_loss + config.ranking_weight * ranking_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach())
            n_batches += 1
        dev_examples, dev_scores = predict(model, dev_loader, device)
        dev_metrics = evaluate(dev_examples, dev_scores, k_values, bootstrap=0, seed=config.seed)
        score = float(dev_metrics["admission_macro"]["auprc"]["mean"])
        history.append({"epoch": epoch, "loss": running_loss / n_batches, "dev_macro_auprc": score})
        print(f"epoch={epoch:03d} loss={running_loss/n_batches:.5f} dev_macro_AUPRC={score:.4f}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    dev_examples, dev_scores = predict(model, dev_loader, device)
    output: Dict[str, Any] = {
        "task": "fine-tuned discharge-summary sentence prediction",
        "config": asdict(config),
        "train": train_stats,
        "dev": dev_stats,
        "test": test_stats,
        "best_epoch": best_epoch,
        "history": history,
        "validation": evaluate(dev_examples, dev_scores, k_values, args.bootstrap, config.seed),
    }
    if test_loader:
        test_examples, test_scores = predict(model, test_loader, device)
        output["test_metrics"] = evaluate(
            test_examples, test_scores, k_values, args.bootstrap, config.seed
        )
        if args.test_predictions:
            write_predictions(args.test_predictions, test_examples, test_scores)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(config), "state_dict": model.state_dict()},
        args.checkpoint,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved checkpoint -> {args.checkpoint}")
    print(f"Saved results -> {args.output_json}")


if __name__ == "__main__":
    main()
