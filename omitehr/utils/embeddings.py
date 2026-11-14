#!/usr/bin/env python3
"""Create reusable sentence embeddings for model training and deduplication.

The module encodes source sentences with a Hugging Face encoder and writes
JSONL plus aligned NPZ arrays for downstream models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from omitehr.utils.entailment import fingerprint

DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


def public_model_identifier(model_name: str) -> str:
    """Avoid embedding a local user's filesystem path in released artifacts."""
    if model_name.startswith(("/", "./", "../", "~")):
        return Path(model_name).name
    return model_name


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Averages token embeddings, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def max_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Same idea as mean_pool, but takes the max instead of the average
    (padding gets masked out so it can't win)."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state.masked_fill(mask == 0, float("-inf"))
    return masked.max(dim=1).values


POOLING_FNS = {"mean": mean_pool, "max": max_pool}


def embed_sentences(
    sentences: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
    max_length: int = 512,
    pooling: str = "mean",
    on_batch_done=None,
) -> List[List[float]]:
    """Embeds a flat list of sentences in batches. Caller should
    flatten across documents first, so batches don't end up tiny.

    on_batch_done, if given, runs after each batch -- handy for writing
    results to disk as you go instead of waiting for everything."""
    if pooling not in POOLING_FNS:
        raise ValueError(f"Unknown pooling {pooling!r} (expected one of {list(POOLING_FNS)})")
    pool_fn = POOLING_FNS[pooling]

    embeddings: List[List[float]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            pooled = pool_fn(output.last_hidden_state, encoded["attention_mask"])
            batch_embeddings = pooled.cpu().tolist()
            embeddings.extend(batch_embeddings)
            if on_batch_done is not None:
                on_batch_done(start, batch_embeddings)
    return embeddings


def write_npz_from_jsonl(
    output_path: Path,
    model_name: str,
    pooling: str,
    max_length: int,
    current_labels: Optional[Dict[Tuple[Any, int], Optional[int]]] = None,
) -> None:
    """Rebuilds the companion .npz from whatever's in output_path right
    now, so it's always complete no matter how many times we resumed."""
    all_rows = load_jsonl(output_path)
    npz_path = output_path.with_suffix(".npz")
    resolved_labels = []
    for row in all_rows:
        label = row["label"]
        if current_labels is not None:
            label = current_labels.get((row["doc_id"], row["sentence_idx"]), label)
        resolved_labels.append(label)
    np.savez(
        npz_path,
        embeddings=np.array([row["embedding"] for row in all_rows], dtype=np.float32),
        labels=np.array(
            [np.nan if label is None else float(label) for label in resolved_labels],
            dtype=np.float32,
        ),
        doc_ids=np.array([row["doc_id"] for row in all_rows]),
        sentence_idx=np.array([row["sentence_idx"] for row in all_rows], dtype=np.int32),
        # Scalar provenance fields let downstream experiment runners verify
        # that train/evaluation files came from the same encoder configuration.
        encoder_name=np.array(public_model_identifier(model_name)),
        pooling=np.array(pooling),
        max_length=np.array(max_length, dtype=np.int32),
    )
    n_labeled = sum(label is not None for label in resolved_labels)
    print(f"Wrote companion arrays ({len(all_rows)} sentences, {n_labeled} labeled) -> {npz_path}")


def load_model(model_name: str, device: Optional[str] = None):
    """Loads the tokenizer + model once so callers can reuse them
    across multiple files instead of reloading each time."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded. Loading model weights: {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    # Transformers enables ModernBERT's optional torch.compile reference path
    # whenever Triton is importable.  Some environments have Triton present but
    # cannot compile its runtime C extension, so explicitly use the normal eager
    # path.  This affects performance only, not model outputs.
    if getattr(model.config, "model_type", None) == "modernbert":
        model.config.reference_compile = False
        print("Disabled ModernBERT reference_compile (using eager path).")
    print(f"Model loaded. Moving to {device}...")
    model = model.to(device)
    print(f"Model ready on {device}.")
    return tokenizer, model, device


def embed_jsonl(
    input_path: Path,
    output_path: Path,
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
    pooling: str = "mean",
    max_length: int = 512,
    model_name: str = DEFAULT_MODEL,
    resume: bool = False,
    checkpoint_every: int = 20,
) -> None:

    docs = load_jsonl(input_path)

    # Flatten (doc_id, sentence_idx, sentence, label) across all docs so
    # batches are full-size instead of one small batch per doc. The official
    # BHC pipeline supplies contextual_label_values, projected from the
    # structured contextual_chunk_labels. Legacy entailment_labels remain a
    # backward-compatible fallback.
    flat_ids: List[Tuple[Any, int]] = []
    flat_sentences: List[str] = []
    flat_labels: List[Optional[int]] = []
    for doc in docs:
        labels = doc.get("contextual_label_values", doc.get("entailment_labels"))
        if labels is not None and len(labels) != len(doc["source_sentences_valid"]):
            raise ValueError(f"Unaligned source chunks/labels for doc_id={doc['doc_id']}")
        for i, sentence in enumerate(doc["source_sentences_valid"]):
            flat_ids.append((doc["doc_id"], i))
            flat_sentences.append(sentence)
            flat_labels.append(labels[i] if labels is not None else None)

    if len(set(flat_ids)) != len(flat_ids):
        raise ValueError("input contains duplicate (doc_id, sentence_idx) pairs")
    provenance_path = output_path.with_suffix(".meta.json")
    provenance = {"input_sha256": fingerprint(flat_ids, flat_sentences, model_name, pooling, max_length)}
    if resume and output_path.exists():
        if not provenance_path.exists() or json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Embedding inputs or configuration changed; rerun without --resume")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Clear old rows before recording new provenance, including for empty input.
    if not resume or not output_path.exists():
        output_path.write_text("")
    provenance_path.write_text(json.dumps(provenance) + "\n")

    done_keys = set()
    if resume and output_path.exists():
        for row in load_jsonl(output_path):
            done_keys.add((row["doc_id"], row["sentence_idx"]))
        print(f"Resuming: {len(done_keys)} sentences already embedded, skipping them.")

    pending_idx = [j for j, key in enumerate(flat_ids) if key not in done_keys]

    if not pending_idx:
        print("Nothing to do -- all sentences already embedded.")
    else:
        pending_sentences = [flat_sentences[j] for j in pending_idx]
        n_total_batches = (len(pending_sentences) + batch_size - 1) // batch_size
        print(f"Embedding {len(pending_sentences)} sentences "
              f"({len(done_keys)} already done, {len(docs)} documents total)...")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(output_path, "a" if resume else "w")
        batch_count = 0

        def on_batch_done(start: int, batch_embeddings: List[List[float]]) -> None:
            nonlocal batch_count
            for k, embedding in enumerate(batch_embeddings):
                j = pending_idx[start + k]
                doc_id, sentence_idx = flat_ids[j]
                f.write(json.dumps({
                    "doc_id": doc_id,
                    "sentence_idx": sentence_idx,
                    "sentence": flat_sentences[j],
                    "label": flat_labels[j],
                    "embedding": embedding,
                }) + "\n")
            f.flush()
            batch_count += 1
            done_now = start + len(batch_embeddings)
            if batch_count % checkpoint_every == 0 or done_now >= len(pending_sentences):
                print(f"Checkpoint: {done_now}/{len(pending_sentences)} pending "
                      f"sentences embedded (batch {batch_count}/{n_total_batches})")

        try:
            embed_sentences(
                pending_sentences,
                tokenizer,
                model,
                device,
                batch_size,
                max_length=max_length,
                pooling=pooling,
                on_batch_done=on_batch_done,
            )
        finally:
            f.close()

        print(f"Wrote {len(pending_sentences)} new sentence embeddings -> {output_path}")

    # Always rebuilt from the full current file (old + new rows), so the
    # .npz stays complete and consistent no matter how many resumes happened.
    if output_path.exists():
        current_labels = dict(zip(flat_ids, flat_labels))
        write_npz_from_jsonl(
            output_path,
            model_name=model_name,
            pooling=pooling,
            max_length=max_length,
            current_labels=current_labels,
        )
    else:
        print("No output file exists yet (input had zero sentences) -- skipping .npz.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed each valid source sentence with ClinicalBERT for "
                    "later classifier training against entailment labels."
    )
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Split+filtered docs from note_sentence_splitter.py. "
                            "Comma-separate multiple files (e.g. train.jsonl,test.jsonl) "
                            "to process them all in one process, loading the model once.")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Same count/order as --input_jsonl, comma-separated")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model id")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum tokens per sentence (default: 512)")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"],
                        help="How each sentence's per-token BERT outputs collapse into "
                            "one vector: mean (default, averages real tokens) or max "
                            "(takes the max activation per dimension across real tokens).")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda/cpu; auto-detected (prefers cuda) if not set")
    parser.add_argument("--resume", action="store_true",
                        help="Append to each --output_jsonl, skipping (doc_id, sentence_idx) "
                            "pairs already present there instead of starting over")
    parser.add_argument("--checkpoint_every", type=int, default=20,
                        help="Print progress every N batches (results are flushed "
                            "to disk after every batch regardless)")
    args = parser.parse_args()

    input_paths = [Path(p.strip()) for p in args.input_jsonl.split(",")]
    output_paths = [Path(p.strip()) for p in args.output_jsonl.split(",")]
    if len(input_paths) != len(output_paths):
        parser.error("--input_jsonl and --output_jsonl must have the same "
                    "number of comma-separated paths")

    tokenizer, model, device = load_model(args.model, args.device)

    for input_path, output_path in zip(input_paths, output_paths):
        print(f"\n=== Embedding {input_path} -> {output_path} (pooling={args.pooling}) ===")
        embed_jsonl(
            input_path, output_path, tokenizer, model, device,
            batch_size=args.batch_size, pooling=args.pooling,
            max_length=args.max_length, model_name=args.model, resume=args.resume,
            checkpoint_every=args.checkpoint_every,
        )


if __name__ == "__main__":
    main()
