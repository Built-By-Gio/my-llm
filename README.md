# my-llm

A small character-level GPT language model written in PyTorch and trained from scratch on Tiny Shakespeare. Each character is one token. The model is a decoder-only transformer (6 layers, 6 attention heads, 384-dimensional embeddings, 256-character context, about 10.8 million parameters) that learns to predict the next character, then writes new text in the style of its training data.

Built by Gio ([Built-By-Gio](https://github.com/Built-By-Gio)).

## Status (2026-11-25)

Working:
- The model, training script and generation script are written.
- I trained it on 2026-10-01 on my PC. The checkpoint from that run (about 43 MB) is not in this repo because of its size.

I did not save loss numbers or generated text samples from that run, so none are shown here.

## What you need

- Python 3 (used with Python 3.13)
- PyTorch 2.11.0 (see `requirements.txt`). My setup used the CUDA 12.8 build (`2.11.0+cu128`) from the PyTorch package index. An NVIDIA GPU is optional; the scripts fall back to the CPU, which is much slower.

## How to run

1. Clone this repo and open a terminal in its folder.
2. Create a virtual environment and activate it:
   `python -m venv .venv` then `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux).
3. Install PyTorch: `pip install -r requirements.txt`. For a GPU build, use the install command for your system from [pytorch.org](https://pytorch.org/get-started/locally/) instead.
4. Train: `python train.py` (default 5000 iterations, batch 64). Options: `--iters N`, `--batch N`. It prints train and validation loss every 250 iterations and saves the best checkpoint to `ckpt.pt`.
5. Generate: `python generate.py "ROMEO:"`. Options: `--tokens N` (default 500) and `--temp T` (default 0.8). Characters in the prompt that never appeared in training are dropped.

To train on your own text, replace `data/input.txt` with any plain-text file before step 4. If `data/input.txt` is missing, `train.py` downloads Tiny Shakespeare.

## Repo layout

```
gpt.py           the model: causal self-attention, transformer blocks, GPT, sampling
train.py         character tokenizer, 90/10 train/validation split, AdamW training loop, checkpointing
generate.py      loads ckpt.pt and samples text from a prompt
data/input.txt   Tiny Shakespeare training text (about 1.1 MB)
requirements.txt pinned PyTorch version
```

## Roadmap / future plans

- Train on my own text corpus instead of Shakespeare.
- Replace the character tokenizer with a BPE (subword) tokenizer.
- Scale the model up, or fine-tune an existing open model instead of training from scratch.

## Credits and license

- The approach follows Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out" lecture and his [nanoGPT](https://github.com/karpathy/nanoGPT) project: a character-level GPT on Tiny Shakespeare with the same default model size.
- Dataset: Tiny Shakespeare, a collection of Shakespeare's works (public domain), as published by Andrej Karpathy in the [char-rnn](https://github.com/karpathy/char-rnn) repository (MIT License).

This project's code is released under the MIT License. See `LICENSE`.
