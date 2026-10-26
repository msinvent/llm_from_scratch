"""Train the GPT model on token files written by prepare_data.py.

    python gpt2/train.py --data tinystories --preset stories     # the main 5080 run
    python gpt2/train.py --data verdict --preset tiny             # quick check that everything works
    python gpt2/train.py --data tinystories --preset stories --resume   # continues the latest tinystories-stories run

Each run gets its own folder, runs/<data>-<preset>-<YYYYmmdd-HHMMSS>, with config.json, log.csv, ckpt.pt and
TensorBoard logs. Compare runs with: tensorboard --logdir runs
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import tiktoken
import torch
from torch.utils.tensorboard import SummaryWriter

from model import GPT, GPTConfig

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RUNS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# batch_tokens = tokens per optimizer step; micro_batch = sequences per forward pass, sized to fit in 16 GB
PRESETS = {
    "tiny":    dict(n_layers=4,  n_heads=4,  d_model=256, context_length=256,  batch_tokens=16_384,  micro_batch=16, lr=1e-3),
    "stories": dict(n_layers=8,  n_heads=8,  d_model=512, context_length=512,  batch_tokens=131_072, micro_batch=32, lr=1e-3),
    "gpt2":    dict(n_layers=12, n_heads=12, d_model=768, context_length=1024, batch_tokens=524_288, micro_batch=16, lr=6e-4),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder under data/ with train.bin and val.bin")
    p.add_argument("--preset", choices=PRESETS, default="stories")
    p.add_argument("--micro-batch", type=int, help="lower this if you run out of GPU memory")
    p.add_argument("--max-steps", type=int, help="default: one pass over the training tokens")
    p.add_argument("--lr", type=float)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--out-dir", help="default: a new runs/<data>-<preset>-<timestamp>")
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/ckpt.pt (default: the latest matching run)")
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile (it is only used on CUDA)")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # print each line right away, even when piped into tee
    args = parse_args()
    preset = PRESETS[args.preset]
    micro_batch = args.micro_batch or preset["micro_batch"]
    lr = args.lr or preset["lr"]
    T = preset["context_length"]
    run_prefix = os.path.join(RUNS_ROOT, f"{args.data}-{args.preset}")
    if args.out_dir:
        out_dir = args.out_dir
    elif args.resume:
        previous = sorted(glob.glob(f"{run_prefix}-[0-9]*-[0-9]*"))  # timestamps sort in time order
        if not previous:
            raise SystemExit(f"no run to resume matching {run_prefix}-*; pass --out-dir")
        out_dir = previous[-1]
    else:
        out_dir = f"{run_prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"run folder {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on NVIDIA GPUs
    # bfloat16 on CUDA roughly doubles speed and halves activation memory; Mac and CPU stay in float32
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    data_dir = os.path.join(DATA_ROOT, args.data)
    n_train = len(np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"))

    def get_batch(split: str):
        # Re-open the memmap on every call so its pages don't pile up in memory over a long run
        data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        starts = torch.randint(len(data) - T - 1, (micro_batch,)).tolist()
        x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in starts])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in starts])  # shifted by one
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    assert preset["batch_tokens"] % (micro_batch * T) == 0, "batch_tokens must be a multiple of micro_batch * context_length"
    grad_accum = preset["batch_tokens"] // (micro_batch * T)  # micro-batches summed into one optimizer step
    max_steps = args.max_steps or max(1, n_train // preset["batch_tokens"])
    warmup_steps = max(1, max_steps // 25)
    min_lr = lr / 10

    def get_lr(step: int) -> float:
        # Linear warmup, then cosine decay down to min_lr
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (lr - min_lr)

    cfg = GPTConfig(**{k: preset[k] for k in ("n_layers", "n_heads", "d_model", "context_length")}, dropout=args.dropout)
    model = GPT(cfg).to(device)
    # Weight decay only on the matrices; biases and LayerNorm weights (1-D) are left alone
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": args.weight_decay},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=device == "cuda",
    )

    step = 0
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if args.resume:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resumed from {ckpt_path} at step {step}")

    raw_model = model  # uncompiled, for saving and generation
    if device == "cuda" and not args.no_compile:
        model = torch.compile(model)

    print(f"device {device} | {raw_model.num_params() / 1e6:.1f}M parameters | {n_train / 1e6:.1f}M training tokens")
    print(f"{max_steps} steps x {preset['batch_tokens']:,} tokens "
          f"(micro-batch {micro_batch} x {T} tokens x {grad_accum} accumulation steps)")

    enc = tiktoken.get_encoding("gpt2")
    log_path = os.path.join(out_dir, "log.csv")
    if not args.resume or not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("step,lr,train_loss,val_loss,tokens_per_sec\n")
        # Everything that defines the run, so runs can be compared later
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"args": vars(args), "preset": preset, "micro_batch": micro_batch, "lr": lr,
                       "max_steps": max_steps, "n_train_tokens": n_train}, f, indent=2)
    tb = SummaryWriter(out_dir)  # writes to disk in a background thread

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(args.eval_batches):
                x, y = get_batch(split)
                with autocast:
                    losses.append(model(x, y)[1].item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    def evaluate_and_save():
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        with open(log_path, "a") as f:
            f.write(f"{step},,{losses['train']:.4f},{losses['val']:.4f},\n")
        tb.add_scalar("loss/val", losses["val"], step)
        tb.add_scalar("loss/train_eval", losses["train"], step)
        prompt = torch.tensor([enc.encode(args.prompt)], device=device)
        raw_model.eval()
        sample = raw_model.generate(prompt, max_new_tokens=100, n_vocab=enc.n_vocab)
        raw_model.train()
        text = enc.decode(sample[0].tolist())
        print("sample:", text.replace("\n", " "))
        tb.add_text("sample", text, step)
        if step > 0:
            torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "config": asdict(cfg), "args": vars(args), "val_loss": losses["val"]}, ckpt_path)

    t0 = time.time()
    while step < max_steps:
        if step % args.eval_interval == 0:
            evaluate_and_save()
            t0 = time.time()  # don't count evaluation time in tokens/sec

        for group in optimizer.param_groups:
            group["lr"] = get_lr(step)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(grad_accum):
            x, y = get_batch("train")
            with autocast:
                _, loss = model(x, y)
            loss = loss / grad_accum  # so the summed gradients are the average over the whole batch
            loss_sum += loss.detach()
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % args.log_interval == 0:
            loss_val = float(loss_sum)  # waits for the GPU, so the timing below is real
            dt = (time.time() - t0) / args.log_interval
            tok_per_sec = preset["batch_tokens"] / dt
            print(f"step {step}/{max_steps} | loss {loss_val:.4f} | lr {get_lr(step - 1):.2e} | "
                  f"grad norm {grad_norm:.2f} | {dt * 1000:.0f} ms/step | {tok_per_sec:,.0f} tok/s")
            with open(log_path, "a") as f:
                f.write(f"{step},{get_lr(step - 1):.6e},{loss_val:.4f},,{tok_per_sec:.0f}\n")
            tb.add_scalar("loss/train_step", loss_val, step)
            tb.add_scalar("lr", get_lr(step - 1), step)
            tb.add_scalar("tokens_per_sec", tok_per_sec, step)
            t0 = time.time()

    evaluate_and_save()
    tb.close()


if __name__ == "__main__":
    main()
