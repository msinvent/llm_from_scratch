"""Generate text from a trained checkpoint.

    python gpt2/sample.py runs/tinystories-stories/ckpt.pt --prompt "Once upon a time" --num-samples 3
    python gpt2/sample.py runs/tinystories-stories/ckpt.pt --prompt "Lily went to the park with her mom and dog" --max-new-words 10 --top-k 50 --temperature 0.8
"""
import argparse

import tiktoken
import torch

from model import GPT, GPTConfig


def generate_words(model: GPT, prompt: torch.Tensor, n_words: int, temperature: float, top_k: int,
                   enc: tiktoken.Encoding) -> str:
    """Sample one token at a time until the continuation holds n_words complete words, and return those words."""
    idx = prompt
    words = []
    for _ in range(n_words * 10):  # safety cap on tokens
        idx = model.generate(idx, 1, temperature, top_k, n_vocab=enc.n_vocab)
        if idx[0, -1].item() == enc.eot_token:  # the model ended the story
            break
        words = enc.decode(idx[0, prompt.shape[1]:].tolist()).split()
        if len(words) > n_words:  # word n_words + 1 has started, so word n_words is complete
            break
    return " ".join(words[:n_words])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=200, help="how many tokens to generate after the prompt")
    p.add_argument("--max-new-words", type=int, help="generate this many words instead (overrides --max-new-tokens)")
    p.add_argument("--temperature", type=float, default=0.8, help="lower = safer, more repetitive text")
    p.add_argument("--top-k", type=int, default=50, help="sample only from the k most likely tokens; 1 = always the most likely")
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
        if args.max_new_words:
            print(args.prompt, generate_words(model, prompt, args.max_new_words, args.temperature, args.top_k, enc))
        else:
            out = model.generate(prompt, args.max_new_tokens, args.temperature, args.top_k, n_vocab=enc.n_vocab)
            print(enc.decode(out[0].tolist()))
        print("-" * 80)


if __name__ == "__main__":
    main()
