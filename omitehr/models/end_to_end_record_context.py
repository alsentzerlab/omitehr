#!/usr/bin/env python3
"""Admission-level salience model with a partially fine-tuned encoder.

Source sentences are encoded, pooled, and scored jointly within each
admission by the record-context transformer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import random

from omitehr.models.record_context import (
    Record,
    RecordContextTransformer,
    listwise_loss,
    masked_bce,
    predict,
)
from omitehr.utils.evaluation import Example, evaluate, add_label_argument
from omitehr.utils.metadata import extract_document_metadata


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_text_records(
    path: Path, chunk_size: int, chunked_blocks: bool = False,
    require_labels: bool = True,
    positive_label: int = 1,
) -> list[Record]:
    """Load admissions as either sampled records or contiguous blocks.

    Training/evaluation requires aligned labels.  Inference can set
    ``require_labels=False``; in that mode all source sentences are retained
    and placeholder labels are attached because the model's forward path only
    needs them for training loss and evaluation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    records = []
    with path.open() as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = extract_document_metadata(row, local_window=0)
            # The established filtered-matched pipeline stores labels as
            # ``entailment_labels``.  The BHC-IM contextual-labeling pipeline
            # stores the aligned 0/1/2 values as ``contextual_label_values``.
            # Supporting both keeps the admission-level model independent of
            # which labeling workflow produced the source chunks.
            labels = row.get("contextual_label_values", row.get("entailment_labels", []))
            has_labels = isinstance(labels, list) and len(labels) == len(metadata.sentence_text)
            if require_labels and not has_labels:
                raise ValueError(f"Unaligned labels for doc_id={row['doc_id']}")
            if not require_labels:
                labels = [0] * len(metadata.sentence_text)
            selected = (
                list(range(len(metadata.sentence_text)))
                if not require_labels
                else [i for i, label in enumerate(labels) if label is not None]
            )
            if not selected:
                continue
            if not chunked_blocks and len(selected) > chunk_size:
                positions = np.linspace(0, len(selected) - 1, chunk_size, dtype=np.int64)
                selected = [selected[int(position)] for position in positions]
            if chunked_blocks:
                n_blocks = math.ceil(len(selected) / chunk_size)
                blocks = np.array_split(np.asarray(selected, dtype=np.int64), n_blocks)
            else:
                blocks = [np.asarray(selected, dtype=np.int64)]
            for indices in blocks:
                indices = np.asarray(indices, dtype=np.int64)
                records.append(
                    Record(
                        doc_id=str(row["doc_id"]),
                        embeddings=np.zeros((len(indices), 1), dtype=np.float32),
                        numeric=metadata.numeric[indices],
                        title_ids=metadata.note_title_bucket[indices],
                        section_ids=metadata.section_id[indices],
                        labels=np.asarray(
                            [int(labels[i] >= positive_label) for i in indices], dtype=np.float32
                        ),
                        sentence_indices=indices,
                        texts=tuple(metadata.sentence_text[i] for i in indices),
                    )
                )
    return records


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).type_as(hidden)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def write_predictions(path: Path, examples: list[Example], scores: list[float]) -> None:
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


class EndToEndModel(nn.Module):
    def __init__(self, model_name: str, context_dim: int, layers: int, heads: int, max_positions: int, unfreeze_last_n: int, init_encoder_checkpoint: Path | None = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        if getattr(self.encoder.config, "model_type", None) == "modernbert":
            self.encoder.config.reference_compile = False
        if init_encoder_checkpoint is not None:
            payload = torch.load(init_encoder_checkpoint, map_location="cpu")
            state = payload.get("state_dict", payload)
            encoder_state = {key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")}
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            if missing:
                print(f"initialized encoder with {len(missing)} missing keys", flush=True)
            if unexpected:
                print(f"initialized encoder with {len(unexpected)} unexpected keys", flush=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = unfreeze_last_n < 0
        if unfreeze_last_n > 0:
            container = getattr(self.encoder, "encoder", None)
            blocks = getattr(container, "layer", None) or getattr(self.encoder, "layers", None)
            if blocks is None:
                raise ValueError("Encoder has no recognized transformer block list")
            for block in blocks[-unfreeze_last_n:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        self.context = RecordContextTransformer(
            int(self.encoder.config.hidden_size), context_dim, layers, heads,
            max_positions=max_positions,
        )

    def encode_record(self, record: Record, device: torch.device, max_length: int, chunk_size: int) -> torch.Tensor:
        vectors = []
        for start in range(0, len(record.texts), chunk_size):
            tokens = self.tokenizer(
                list(record.texts[start:start + chunk_size]), padding=True,
                truncation=True, max_length=max_length, return_tensors="pt",
            ).to(device)
            output = self.encoder(**tokens).last_hidden_state
            vectors.append(mean_pool(output, tokens["attention_mask"]))
        return torch.cat(vectors, dim=0)

    def forward_record(self, record: Record, device: torch.device, max_length: int, chunk_size: int) -> torch.Tensor:
        embeddings = self.encode_record(record, device, max_length, chunk_size)
        batch = {
            "embeddings": embeddings[None, :, :],
            "numeric": torch.from_numpy(record.numeric).to(device)[None, :, :],
            "title_ids": torch.from_numpy(record.title_ids).to(device)[None, :],
            "section_ids": torch.from_numpy(record.section_ids).to(device)[None, :],
            "mask": torch.ones((1, len(record.labels)), dtype=torch.bool, device=device),
        }
        return self.context(batch).squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_label_argument(parser)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--dev_jsonl", type=Path, required=True)
    parser.add_argument("--test_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init_encoder_checkpoint", type=Path)
    parser.add_argument("--test_predictions", type=Path)
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help="Deprecated alias: use --train_max_sentences and --eval_max_sentences.",
    )
    parser.add_argument(
        "--train_max_sentences",
        type=int,
        default=1024,
        help="Contiguous training-block size (default: 1024).",
    )
    parser.add_argument(
        "--eval_max_sentences",
        type=int,
        default=1024,
        help="Contiguous evaluation-block size (default: 1024).",
    )
    parser.add_argument(
        "--position_capacity",
        type=int,
        default=1024,
        help="Admission positional-embedding capacity (default: 1024).",
    )
    parser.add_argument(
        "--chunked_blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain all labeled sentences and score contiguous blocks (default: enabled).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute encoder activations during backward to reduce GPU memory (default: enabled).",
    )
    parser.add_argument("--encoder_max_length", type=int, default=256)
    parser.add_argument("--encoder_chunk_size", type=int, default=32)
    parser.add_argument("--context_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--head_learning_rate", type=float, default=3e-4)
    parser.add_argument("--unfreeze_last_n", type=int, default=8)
    parser.add_argument("--ranking_weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--wandb_project",
        help="Enable Weights & Biases logging under this project name.",
    )
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_run_name")
    parser.add_argument("--wandb_group")
    parser.add_argument("--wandb_dir", type=Path)
    parser.add_argument(
        "--epoch_log_jsonl",
        type=Path,
        help="Append each completed epoch's metrics for recovery or later sync.",
    )
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args()
    if args.train_max_sentences is None:
        args.train_max_sentences = args.max_sentences if args.max_sentences is not None else 512
    if args.eval_max_sentences is None:
        args.eval_max_sentences = args.train_max_sentences
    if args.train_max_sentences <= 0 or args.eval_max_sentences <= 0:
        raise ValueError("Sentence block sizes must be positive")
    if args.position_capacity is None:
        args.position_capacity = args.train_max_sentences
    if args.position_capacity <= 0:
        raise ValueError("--position_capacity must be positive")
    if args.chunked_blocks and args.eval_max_sentences != args.train_max_sentences:
        raise ValueError(
            "--eval_max_sentences must equal --train_max_sentences for block evaluation"
        )
    if args.chunked_blocks and args.position_capacity != args.train_max_sentences:
        raise ValueError(
            "--position_capacity must equal --train_max_sentences for block evaluation"
        )
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "--wandb_project requires the wandb package in the training environment"
            ) from error
        if args.wandb_dir:
            args.wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            dir=str(args.wandb_dir) if args.wandb_dir else None,
            mode=args.wandb_mode,
            config=wandb_config,
        )
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train = load_text_records(args.train_jsonl, args.train_max_sentences, args.chunked_blocks, positive_label=args.positive_label)
    # Keep validation aligned with test evaluation so early stopping is not
    # selected on a different sentence population for each training cap.
    dev = load_text_records(args.dev_jsonl, args.eval_max_sentences, args.chunked_blocks, positive_label=args.positive_label)
    test = load_text_records(args.test_jsonl, args.eval_max_sentences, args.chunked_blocks, positive_label=args.positive_label)
    if wandb_run:
        wandb_run.config.update(
            {
                "train_block_count": len(train),
                "dev_block_count": len(dev),
                "test_block_count": len(test),
                "train_admission_count": len({record.doc_id for record in train}),
                "dev_admission_count": len({record.doc_id for record in dev}),
                "test_admission_count": len({record.doc_id for record in test}),
            },
            allow_val_change=True,
        )
    model = EndToEndModel(args.model_name, args.context_dim, args.layers, args.heads, args.position_capacity, args.unfreeze_last_n, args.init_encoder_checkpoint)
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = model.to(device)
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder_parameters}
    head_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_parameters, "lr": args.learning_rate}, {"params": head_parameters, "lr": args.head_learning_rate}],
        weight_decay=1e-2,
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score, best_state, best_epoch, stale = -math.inf, None, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        epoch_train = list(train)
        random.Random(args.seed + epoch).shuffle(epoch_train)
        for index, record in enumerate(epoch_train, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model.forward_record(record, device, args.encoder_max_length, args.encoder_chunk_size)
                labels = torch.from_numpy(record.labels).to(device)
                mask = torch.ones((1, len(labels)), dtype=torch.bool, device=device)
                loss = masked_bce(logits[None, :], labels[None, :], mask)
                if args.ranking_weight:
                    loss = loss + args.ranking_weight * listwise_loss(logits[None, :], labels[None, :], mask)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            total += float(loss.detach())
            if index % 25 == 0: print(f"epoch={epoch} admission={index}/{len(train)}", flush=True)
        # For the first implementation, validation uses the same streaming path.
        model.eval(); examples=[]; scores=[]
        with torch.no_grad():
            for record in dev:
                logits = model.forward_record(record, device, args.encoder_max_length, args.encoder_chunk_size)
                examples.extend(Example(record.doc_id, int(i), "", int(y)) for i, y in zip(record.sentence_indices, record.labels))
                scores.extend(torch.sigmoid(logits).cpu().tolist())
        metrics = evaluate(examples, scores, [5, 10, 20, 50], bootstrap=0, seed=args.seed)
        dev_score = float(metrics["admission_macro"]["auprc"]["mean"])
        epoch_loss = total / len(train)
        history.append({"epoch": epoch, "loss": epoch_loss, "dev_macro_auprc": dev_score})
        print(f"epoch={epoch} loss={epoch_loss:.5f} dev_macro_AUPRC={dev_score:.4f}", flush=True)
        epoch_metrics = {
            "epoch": epoch,
            "train/loss": epoch_loss,
            "dev/admission_macro_auprc": dev_score,
            "dev/best_admission_macro_auprc": max(best_score, dev_score),
        }
        if args.epoch_log_jsonl:
            args.epoch_log_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.epoch_log_jsonl.open("a") as epoch_log:
                epoch_log.write(json.dumps(epoch_metrics) + "\n")
        if wandb_run:
            wandb_run.log(epoch_metrics, step=epoch, commit=True)
            wandb_run.summary.update(epoch_metrics)
            print(
                f"W&B logged epoch={epoch} run={wandb_run.id}",
                flush=True,
            )
        if dev_score > best_score:
            best_score, best_epoch, stale = dev_score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("No checkpoint produced")
    model.load_state_dict(best_state); model.eval(); examples=[]; scores=[]
    with torch.no_grad():
        for record in test:
            logits = model.forward_record(record, device, args.encoder_max_length, args.encoder_chunk_size)
            examples.extend(Example(record.doc_id, int(i), "", int(y)) for i, y in zip(record.sentence_indices, record.labels))
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    test_metrics = evaluate(
        examples, scores, [5, 10, 20, 50], bootstrap=1000, seed=args.seed
    )
    output = {
        "task": "end-to-end admission contextual BCE+listwise",
        "model_name": args.model_name,
        "chunked_blocks": args.chunked_blocks,
        "args": serializable_args,
        "train_block_count": len(train),
        "dev_block_count": len(dev),
        "test_block_count": len(test),
        "train_admission_count": len({record.doc_id for record in train}),
        "dev_admission_count": len({record.doc_id for record in dev}),
        "test_admission_count": len({record.doc_id for record in test}),
        "train_max_sentences": args.train_max_sentences,
        "eval_max_sentences": args.eval_max_sentences,
        "position_capacity": args.position_capacity,
        "best_epoch": best_epoch,
        "history": history,
        "test_metrics": test_metrics,
    }
    if args.test_predictions:
        write_predictions(args.test_predictions, examples, scores)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": serializable_args,
                "state_dict": model.state_dict()}, args.checkpoint)
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(output, indent=2)+"\n")
    if wandb_run:
        wandb_run.summary.update(
            {
                "best_epoch": best_epoch,
                "best_dev/admission_macro_auprc": best_score,
                "test/pooled_auroc": test_metrics["pooled"]["auroc"],
                "test/pooled_auprc": test_metrics["pooled"]["auprc"],
                "test/admission_macro_auroc": test_metrics["admission_macro"]["auroc"]["mean"],
                "test/admission_macro_auprc": test_metrics["admission_macro"]["auprc"]["mean"],
            }
        )
        wandb_run.finish()
    print(f"Saved results -> {args.output_json}")


if __name__ == "__main__": main()
