"""Tokenize a dataset with the GPT-2 tokenizer into data/<name>/train.bin and val.bin (uint16 token ids).

    python gpt2/prepare_data.py tinystories
    python gpt2/prepare_data.py verdict

Skips a dataset whose train.bin and val.bin already exist; pass --force to rebuild them.
"""
import argparse
import os
import urllib.request
from collections.abc import Iterable, Iterator

import numpy as np
import tiktoken
from tqdm import tqdm

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
VERDICT_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"


def batched(texts: Iterable[str], size: int) -> Iterator[list[str]]:
    batch = []
    for text in texts:
        batch.append(text)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_tokens(texts: Iterable[str], path: str, enc: tiktoken.Encoding, max_tokens: int | None = None) -> int:
    """Tokenize texts in batches and write them to path, with <|endoftext|> after each document."""
    n_tokens = 0
    tmp_path = path + ".tmp"  # renamed to path only once complete, so an interrupted run never leaves a partial path
    with open(tmp_path, "wb") as f, tqdm(desc=os.path.basename(path), unit=" tok", unit_scale=True) as bar:
        for batch in batched(texts, 1024):
            docs = enc.encode_ordinary_batch(batch, num_threads=os.cpu_count() or 1)
            ids = np.fromiter((t for doc in docs for t in (*doc, enc.eot_token)), dtype=np.uint16)  # vocab 50257 fits in uint16
            if max_tokens is not None:
                ids = ids[: max_tokens - n_tokens]
            ids.tofile(f)
            n_tokens += len(ids)
            bar.update(len(ids))
            if max_tokens is not None and n_tokens >= max_tokens:
                break
    os.replace(tmp_path, path)
    return n_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["tinystories", "verdict"])
    parser.add_argument("--max-tokens", type=float, default=None, help="cap on training tokens, e.g. 1e7 for a quick test")
    parser.add_argument("--force", action="store_true", help="rebuild train.bin and val.bin even if they already exist")
    args = parser.parse_args()
    max_tokens = int(args.max_tokens) if args.max_tokens else None

    out_dir = os.path.join(DATA_ROOT, args.dataset)
    bin_paths = [os.path.join(out_dir, f"{name}.bin") for name in ("train", "val")]
    if not args.force and all(os.path.exists(p) for p in bin_paths):
        for p in bin_paths:
            print(f"{p}: {os.path.getsize(p) // 2:,} tokens (already exists, skipping; use --force to rebuild)")
        return

    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == "verdict":
        path = os.path.join(out_dir, "the-verdict.txt")
        if not os.path.exists(path):
            urllib.request.urlretrieve(VERDICT_URL, path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        split = int(0.9 * len(text))
        splits = {"train": [text[:split]], "val": [text[split:]]}
    else:
        from datasets import load_dataset
        # Downloads the dataset's parquet files (~1 GB) once into ~/.cache/huggingface; much faster than streaming
        ds = load_dataset("roneneldan/TinyStories")
        splits = {"train": (row["text"] for row in ds["train"]), "val": (row["text"] for row in ds["validation"])}

    for name, texts in splits.items():
        path = os.path.join(out_dir, f"{name}.bin")
        n = write_tokens(texts, path, enc, max_tokens if name == "train" else None)
        print(f"{path}: {n:,} tokens")


if __name__ == "__main__":
    main()
