"""Generate text from the trained model. Usage: python generate.py "ROMEO:" [--tokens 500] [--temp 0.8]"""
import argparse
from pathlib import Path

import torch

from gpt import GPT

parser = argparse.ArgumentParser()
parser.add_argument("prompt", nargs="?", default="\n")
parser.add_argument("--tokens", type=int, default=500)
parser.add_argument("--temp", type=float, default=0.8)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(Path(__file__).parent / "ckpt.pt", map_location=device)
chars = ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}

model = GPT(len(chars), **ckpt["config"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# characters the model never saw in training are dropped from the prompt
idx = torch.tensor([[stoi[c] for c in args.prompt if c in stoi] or [0]], device=device)
out = model.generate(idx, args.tokens, args.temp)[0].tolist()
print("".join(chars[i] for i in out))
