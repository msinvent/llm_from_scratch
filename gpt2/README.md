# GPT-2 training experiment

A GPT-2 style model trained from scratch, built from the same pieces as `attention.ipynb`.

| File | What it does |
|---|---|
| `model.py` | The model: `MultiHeadAttention`, `FeedForward`, `TransformerBlock`, `GPT` |
| `prepare_data.py` | Downloads a dataset and tokenizes it into `data/<name>/train.bin` and `val.bin` |
| `train.py` | Trains the model, evaluates every 250 steps, saves `runs/<data>-<preset>/ckpt.pt` and `log.csv` |
| `sample.py` | Generates text from a checkpoint |

## Model sizes (`--preset`)

| Preset | Layers | Heads | `d_model` | Context | Parameters | Use |
|---|---|---|---|---|---|---|
| `tiny` | 4 | 4 | 256 | 256 | 16M | checking that the code runs |
| `stories` | 8 | 8 | 512 | 512 | 51M | the TinyStories run |
| `gpt2` | 12 | 12 | 768 | 1024 | 124M | GPT-2 small |

## Setup on Windows + WSL2 (RTX 5080)

1. Install the latest NVIDIA driver **on Windows**. Do not install a Linux NVIDIA driver inside WSL; WSL uses the Windows one. Check inside WSL with `nvidia-smi`.
2. Clone the repo **inside the WSL filesystem** (e.g. `~/llm_from_scratch`), not under `/mnt/c/...`, which is much slower for reading the data files.
3. Create the environment. The RTX 50 series needs PyTorch 2.7+ built for CUDA 12.8 or newer:

```bash
sudo apt update && sudo apt install -y build-essential python3-venv   # torch.compile needs a C compiler
cd ~/llm_from_scratch
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The last line should print `True NVIDIA GeForce RTX 5080`.

## Run

```bash
# 1. Quick check (a few seconds): tokenizes the-verdict.txt and trains the tiny model
python gpt2/prepare_data.py verdict
python gpt2/train.py --data verdict --preset tiny --max-steps 50

# 2. Download and tokenize TinyStories (~1 GB download, ~2 minutes, 474M training tokens)
python gpt2/prepare_data.py tinystories

# 3. Train (one pass over the data = 3,616 steps of 131k tokens)
python gpt2/train.py --data tinystories --preset stories

# 4. Generate text
python gpt2/sample.py runs/tinystories-stories/ckpt.pt --prompt "Once upon a time"
```

For long runs, start them inside `tmux` so they keep going if the terminal closes (`tmux new -s train`, detach with `Ctrl-b d`, reattach with `tmux attach -t train`). If a run stops, add `--resume` to continue from the last checkpoint.

## What to expect

- The loss starts at about 10.8, which is $\ln(50304)$: a model that guesses uniformly over the vocabulary.
- On TinyStories, samples become readable English after a few hundred steps, and the validation loss should end up somewhere around 1.3 to 1.5.
- The 5080 should finish the `stories` run in roughly an hour; the first steps are slower while `torch.compile` compiles the model.
- Out of GPU memory: lower `--micro-batch` (e.g. 16 or 8). The gradient accumulation steps go up so each optimizer step still sees the same number of tokens.

## On a Mac

Everything also runs on Apple silicon (`mps`), in float32 and without `torch.compile`. It is fine for the quick check and the `tiny` preset, but a 24 GB MacBook Air does about 4,800 tokens/s on `stories` with `--micro-batch 8` (the default 32 runs out of memory), so a full TinyStories pass would take over a day.
