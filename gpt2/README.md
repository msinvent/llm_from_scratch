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
sudo apt update && sudo apt install -y build-essential python3-dev python3-venv   # torch.compile needs a C compiler and Python headers
gcc --version   # without a compiler, train.py fails with "Failed to find C compiler" (or run it with --no-compile)
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

# Stop here and check the data and the GPU before a long run
ls -lh data/tinystories/   # train.bin ~904 MB, val.bin ~9 MB
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
python gpt2/train.py --data tinystories --preset stories --max-steps 2   # smoke test: loss should drop from ~10.9

# 3. Train (one pass over the data = 3,616 steps of 131k tokens), inside a tmux session named "train"
mkdir -p runs
tmux new -s train "source .venv/bin/activate && python gpt2/train.py --data tinystories --preset stories 2>&1 | tee runs/train.log"

# 4. Generate text
python gpt2/sample.py runs/tinystories-stories/ckpt.pt --prompt "Once upon a time"
python gpt2/sample.py runs/tinystories-stories/ckpt.pt --prompt "Lily went to the park with her mom and dog" --max-new-words 10 --top-k 50 --temperature 0.8
```

The training command runs inside `tmux` so it keeps going if the terminal closes. Detach with `Ctrl-b d`, reattach with `tmux attach -t train`, or follow the output from anywhere with `tail -f runs/train.log`. The tmux session closes when training ends, but the full output stays in `runs/train.log`. If a run stops, launch it again with `--resume` to continue from the last checkpoint (and `tee -a` so the log is appended to, not overwritten).

## What to expect

- The loss starts at about 10.8, which is $\ln(50304)$: a model that guesses uniformly over the vocabulary.
- On TinyStories, samples become readable English after a few hundred steps, and the validation loss should end up somewhere around 1.3 to 1.5.
- The 5080 should finish the `stories` run in roughly an hour; the first steps are slower while `torch.compile` compiles the model.
- Out of GPU memory: lower `--micro-batch` (e.g. 16 or 8). The gradient accumulation steps go up so each optimizer step still sees the same number of tokens.

## Predicting the next words

`sample.py` continues a prompt. The length of the continuation is set in one of two ways:

- `--max-new-tokens N` (default 200): generate exactly N tokens. A token is what the GPT-2 tokenizer splits text into: often a whole word with its leading space (`" park"`), sometimes part of a word or a punctuation mark. Simple English averages about 1.3 tokens per word.
- `--max-new-words N`: keep generating until the continuation has N words, then cut it there. Words are counted by splitting on whitespace, so `"dog."` counts as one word. It stops early if the model ends the story.

How the next token is picked:

- `--top-k K` (default 50): only the K most likely tokens are kept, and the next token is sampled from those. `1` always takes the single most likely token (greedy decoding), so every run gives the same output, but it tends to loop. 10 to 50 gives sensible variety, and several hundred lets in odd, unlikely tokens.
- `--temperature T` (default 0.8): divides the model's scores before they become probabilities. Below 1 (e.g. 0.5) sharpens them, so the model sticks to safe, predictable choices. Above 1 (e.g. 1.2) flattens them, so it takes more risks and the text gets wilder.
- `--num-samples N` (default 3): how many separate continuations to print, useful for comparing settings.

Things to know about the predictions:

- **One token at a time.** The model only ever predicts the next token. Ten words means roughly 13 forward passes, each one feeding the model's own previous guesses back in as input. Predicting several words in a single pass would need a different training objective (multi-token prediction), which is one of this project's research goals.
- **Random by default.** With `--temperature 0.8 --top-k 50`, each run gives a different continuation. `--top-k 1` always picks the most likely token, so the output is the same every time, but it tends to repeat itself.
- **Only knows TinyStories.** It continues simple children's stories well. Prompts about anything else get turned back into a story.

## On a Mac

Everything also runs on Apple silicon (`mps`), in float32 and without `torch.compile`. It is fine for the quick check and the `tiny` preset, but a 24 GB MacBook Air does about 4,800 tokens/s on `stories` with `--micro-batch 8` (the default 32 runs out of memory), so a full TinyStories pass would take over a day.
