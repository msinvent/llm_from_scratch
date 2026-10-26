"""Generate text from a trained checkpoint.

    python gpt2/sample.py runs/tinystories-stories/ckpt.pt --prompt "Once upon a time" --num-samples 3
"""
import argparse

import tiktoken
import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="lower = safer, more repetitive text")
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"step {ckpt['step']}, val loss {ckpt['val_loss']:.4f}\n")

    enc = tiktoken.get_encoding("gpt2")
    prompt = torch.tensor([enc.encode(args.prompt)], device=device)
    for _ in range(args.num_samples):
        out = model.generate(prompt, args.max_new_tokens, args.temperature, args.top_k, n_vocab=enc.n_vocab)
        print(enc.decode(out[0].tolist()))
        print("-" * 80)


if __name__ == "__main__":
    main()
