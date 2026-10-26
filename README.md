LLM From Scratch

This project intend to implement an LLM from scratch following "LLM from scratch" book by Sebastian Rashka and a youtube video series by Vizuara that is heavily based on the book.

The Primary goal of the Project is to:

  1. Learn the nuts and bolts of LLM.
  2. The end goal of this project is to make a personal assistant with 2 Billion Parameters. Trained from scratch on publicly available data and finetuned on requirements and data from some personal use cases.

Research Goals:

  1. Implement a model that does prediction of next words in batches instead of 1 word at a time.



Setup (Ubuntu / WSL2):

1. System packages. `build-essential` gives the C compiler that `torch.compile` (Triton) needs, `python3-dev` the Python headers it compiles against, and `python3-venv` the `venv` module:

```bash
sudo apt update && sudo apt install -y build-essential python3-dev python3-venv
gcc --version   # should print a version, not "command not found"
```

2. NVIDIA driver: on WSL2, install the latest driver on Windows only (no Linux driver inside WSL). `nvidia-smi` inside WSL should list the GPU.

3. Python environment. Install PyTorch from the CUDA 12.8 wheel index first (the RTX 50 series needs PyTorch 2.7+ with CUDA 12.8+), then the rest:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

4. Check that PyTorch sees the GPU and can run on it:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name()); x = torch.randn(1024, 1024, device='cuda'); print((x @ x).device)"
```

This should print `True NVIDIA GeForce RTX 5080` and `cuda:0`. See `gpt2/README.md` for running the training experiment.