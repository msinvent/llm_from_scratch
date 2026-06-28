LLM From Scratch

This project intend to implement an LLM from scratch following "LLM from scratch" book by Sebastian Rashka and a youtube video series by Vizuara that is heavily based on the book.

The Primary goal of the Project is to:

  1. Learn the nuts and bolts of LLM.
  2. The end goal of this project is to make a personal assistant with 2 Billion Parameters. Trained from scratch on publicly available data and finetuned on requirements and data from some personal use cases.

Research Goals:

  1. Implement a model that does brediction of next words in batches instead of 1 word at a time.



Setup:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt