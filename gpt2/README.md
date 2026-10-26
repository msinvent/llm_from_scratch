# GPT-2 training experiment

A GPT-2 style model trained from scratch, built from the same pieces as `attention.ipynb`.

| File | What it does |
|---|---|
| `model.py` | The model: `MultiHeadAttention`, `FeedForward`, `TransformerBlock`, `GPT` |
| `prepare_data.py` | Downloads a dataset and tokenizes it into `data/<name>/train.bin` and `val.bin` |
| `train.py` | Trains the model, evaluates every 250 steps, saves `config.json`, `log.csv`, `ckpt.pt` and TensorBoard logs in a new `runs/<data>-<preset>-<timestamp>/` |
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
tmux new -s train "source .venv/bin/activate && python -u gpt2/train.py --data tinystories --preset stories 2>&1 | tee runs/train.log"

# 4. Generate text
# (use your run's folder name, e.g. runs/tinystories-stories-20261026-020147)
python gpt2/sample.py runs/tinystories-stories-<timestamp>/ckpt.pt --prompt "Once upon a time"
python gpt2/sample.py runs/tinystories-stories-<timestamp>/ckpt.pt --prompt "Lily went to the park with her mom and dog" --max-new-words 10 --top-k 50 --temperature 0.8
```

The training command runs inside `tmux` so it keeps going if the terminal closes. Detach with `Ctrl-b d`, reattach with `tmux attach -t train`, or follow the output from anywhere with `tail -f runs/train.log`. The tmux session closes when training ends, but the full console output stays in `runs/train.log` (overwritten by the next launch; the numbers that matter are also in the run's own `log.csv`). If a run stops, launch it again with `--resume` to continue the latest run of that data and preset from its last checkpoint (and `tee -a` so the console log is appended to, not overwritten).

## Tracking and comparing runs

Every run gets its own folder, so runs never overwrite each other:

```
runs/tinystories-stories-20261026-020147/
  config.json   # all settings of the run: command-line args, preset, learning rate, number of steps
  log.csv       # step, lr, train_loss, val_loss, tokens_per_sec
  ckpt.pt       # model and optimizer, saved at every evaluation
  events.out.tfevents.*   # the same numbers for TensorBoard
  eval_train/, eval_val/  # the eval losses for TensorBoard, one sub-run each so they get their own colors
```

`log.csv` has two kinds of rows. Every 10 steps (`--log-interval`): the loss of that step's batch, the learning rate and the speed, with `val_loss` empty. Every 250 steps (`--eval-interval`): train and val loss each averaged over 20 batches (`--eval-batches`), with `lr` empty.

To see and compare the curves, start TensorBoard in a second terminal (it can stay open while training runs and picks up new points and new runs by itself):

```bash
source .venv/bin/activate
tensorboard --logdir runs        # then open http://localhost:6006 in your Windows browser
```

| Chart | What it is |
|---|---|
| `loss/train_step` | loss of the training batch, every 10 steps (noisy; use the Smoothing slider) |
| `loss/eval` | train and val loss averaged over 20 batches, every 250 steps, on one chart: the `<run>/eval_train` and `<run>/eval_val` lines |
| `lr` | learning rate: linear warmup, then cosine decay |
| `tokens_per_sec` | training speed |

The Smoothing slider (top left, default 0.6) draws the smoothed curve in full color and the raw values as a faint line behind it. Set it to 0 for `loss/eval`, which has only a few points.

The charts update live, 10 to 60 seconds behind the terminal: `train.py` writes to disk every ~40 steps, TensorBoard checks for new data every 5 seconds, and the page refreshes every 30 seconds (click ⟳ at the top right to refresh now). Val points only appear every 250 steps, when an evaluation runs.

Each run folder shows up as its own run (plus its `eval_train` and `eval_val` sub-runs), including new ones started while TensorBoard is open; tick runs on and off in the left panel to compare them.

To look at only some runs, type a regular expression in the **Filter runs** box in the left panel, e.g. `tinystories-stories-.*` for all TinyStories runs or `verdict-tiny-20261026-.*` for the verdict runs of one day. Runs started later that match the filter show up too.

`--logdir` does not accept a wildcard. To load only the folders matching a pattern, let the shell expand it into `--logdir_spec`, which takes a comma-separated list of folders:

```bash
tensorboard --logdir_spec $(ls -d runs/tinystories-stories-* | paste -sd, -)
```

The pattern is expanded once, at startup, so a matching run started later needs a TensorBoard restart. `--logdir_spec` is marked experimental, so prefer the filter box. A val loss that flattens or rises while the train loss keeps falling means the model is overfitting.

To test a change, run the same command with one setting changed (e.g. `--lr 5e-4`) and compare the two runs; `config.json` records what differed.

## What to expect

- The loss starts at about 10.8, which is $\ln(50304)$: a model that guesses uniformly over the vocabulary.
- On TinyStories, samples become readable English after a few hundred steps, and the validation loss should end up somewhere around 1.3 to 1.5.
- The 5080 finishes the `stories` run in about 36 minutes (~200k tokens/s); the first steps are slower while `torch.compile` compiles the model.
- Out of GPU memory: lower `--micro-batch` (e.g. 16 or 8). The gradient accumulation steps go up so each optimizer step still sees the same number of tokens.

## Results and next steps

First full run (`stories` preset, one pass over TinyStories = 3,616 steps, 474M tokens, 36 minutes on the RTX 5080):

| step | train loss | val loss |
|---|---|---|
| 0 | 10.94 | 10.94 |
| 500 | 1.97 | 2.00 |
| 1000 | 1.65 | 1.68 |
| 2000 | 1.45 | 1.48 |
| 3616 | **1.33** | **1.37** |

Both losses fell for the whole run and stayed within ~0.04 of each other, so the model is not overfitting. Val was still falling at the end, so it has more to learn: training longer is the most direct way to a lower loss.

### Training longer

Start a **fresh** run with more steps, e.g. two passes over the data (~75 minutes):

```bash
tmux new -s train "source .venv/bin/activate && python -u gpt2/train.py --data tinystories --preset stories --max-steps 7232 2>&1 | tee runs/train.log"
```

Do not extend a finished run with `--resume --max-steps 7232`. The learning rate schedule (warmup, then cosine decay) is stretched over `max_steps`, so the finished run is already at its minimum learning rate; resuming with a larger `max_steps` recomputes the schedule and jumps the learning rate back up mid-curve (from 1e-4 to about 5.8e-4), which knocks the loss up before it recovers. `--resume` is for continuing an interrupted run with the same settings.

In a longer run, watch the gap between the two `loss/eval` lines: if val flattens while train keeps falling, the model has learned what it can from this data at this size.

### Comparing with other models

There is no single global metric or leaderboard this number fits into:

- **Loss depends on the tokenizer and the validation set.** 1.37 is the average loss per GPT-2 token on TinyStories' validation split. It is directly comparable only with models measured the same way. A tokenizer with a smaller vocabulary has fewer choices per prediction and gives a lower per-token loss for the same quality of model.
- **Bits per byte** (total loss divided by the number of bytes of text, in bits) removes the tokenizer from the comparison, and is the fair way to compare models with different tokenizers on the same text.
- **The TinyStories paper** (Eldan & Li, 2023) did not rank models by loss; it had GPT-4 grade generated stories for grammar, consistency and creativity. Its reference models (e.g. `roneneldan/TinyStories-33M` on Hugging Face) can be evaluated on the same validation set for a head-to-head comparison.
- **General LLM benchmarks** (WikiText perplexity, HellaSwag, MMLU) do not apply: a model that only knows children's stories scores near chance on them.

### Why llama2.c reports lower TinyStories losses

[llama2.c](https://github.com/karpathy/llama2.c) is not more efficient at training: its `train.py` is PyTorch code much like this one (`torch.compile`, bf16, gradient accumulation), and its `run.c` is a small pure-C program for running a trained model on the CPU, not for training. Its lower numbers come from:

1. **Longer training**, on many more tokens than one pass. Probably the biggest factor.
2. **A different tokenizer** (Llama 2's 32k vocabulary, or a 4k one trained on TinyStories), so its per-token losses are not directly comparable with these.
3. **The Llama architecture**: RMSNorm instead of LayerNorm, rotary position embeddings (RoPE) instead of learned ones, SwiGLU feed-forward, no biases. Typically a little better per parameter than the GPT-2 design in `model.py`, but a modest difference.

Next experiments, in rough order of effort: train longer; try the `gpt2` preset (124M parameters, about 3-4x slower per token); add a bits-per-byte evaluation to compare with other models; upgrade `model.py` to Llama-style blocks one piece at a time (RMSNorm, then RoPE, then SwiGLU) and compare each change in TensorBoard.

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
