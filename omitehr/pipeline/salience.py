#!/usr/bin/env python3
"""Time the selected contextual BioClinRoBERTa salience model per admission.

Produces compact JSONL records with ``doc_id``, source-sentence index, score,
and total wall-clock inference time for that admission.  It is intentionally
separate from the LLM audit benchmark so it can run in the PyTorch environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from omitehr.models.end_to_end_record_context import EndToEndModel, load_text_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--model-name", required=True,
        help="Path or Hugging Face identifier for the encoder used by the checkpoint.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--encoder-max-length", type=int)
    parser.add_argument("--encoder-chunk-size", type=int)
    parser.add_argument("--unfreeze-last-n", type=int, default=8)
    parser.add_argument("--max-admissions", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    args.block_size = args.block_size or config.get("eval_max_sentences", 1024)
    args.encoder_max_length = args.encoder_max_length or config.get("encoder_max_length", 256)
    args.encoder_chunk_size = args.encoder_chunk_size or config.get("encoder_chunk_size", 32)
    # BHC generation produces unlabeled records; labels are only needed by
    # the training/evaluation loader, not by the model's forward path.
    records = load_text_records(
        args.input_jsonl, args.block_size, chunked_blocks=True, require_labels=False
    )
    allowed = None
    if args.max_admissions is not None:
        allowed = set()
        for record in records:
            if len(allowed) < args.max_admissions:
                allowed.add(record.doc_id)
        records = [record for record in records if record.doc_id in allowed]
    model = EndToEndModel(
        args.model_name, config.get("context_dim", 256), config.get("layers", 2),
        config.get("heads", 8), config.get("position_capacity", args.block_size), args.unfreeze_last_n,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    results = []
    timing = defaultdict(float)
    with torch.no_grad():
        for record in records:
            started = time.perf_counter()
            scores = torch.sigmoid(model.forward_record(record, device, args.encoder_max_length, args.encoder_chunk_size)).cpu().tolist()
            timing[record.doc_id] += time.perf_counter() - started
            results.extend({"doc_id": record.doc_id, "sentence_idx": int(index), "score": float(score)}
                           for index, score in zip(record.sentence_indices, scores))
    # Add timing once per document, on its first score row, to avoid a separate join file.
    seen = set()
    for row in results:
        if row["doc_id"] not in seen:
            row["salience_seconds"] = timing[row["doc_id"]]
            seen.add(row["doc_id"])
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"admissions": len(timing), "sentences": len(results), "output": str(args.output_jsonl)}, indent=2))


if __name__ == "__main__":
    main()
