"""Train the model on data/input.txt. Usage: python train.py [--iters 5000]"""
import argparse
import time
import urllib.request
from pathlib import Path

import torch

from gpt import GPT

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=5000)
parser.add_argument("--batch", type=int, default=64)
args = parser.parse_args()

root = Path(__file__).parent
data_path = root / "data" / "input.txt"
ckpt_path = root / "ckpt.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
config = dict(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2)

# Data: any plain text file works. Falls back to ~1MB of Shakespeare.
if not data_path.exists():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("No data/input.txt found, downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(url, data_path)
text = data_path.read_text(encoding="utf-8")

# Tokenizer: one token per unique character
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


def get_batch(d):
    ix = torch.randint(len(d) - config["block_size"] - 1, (args.batch,))
    x = torch.stack([d[i : i + config["block_size"]] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + config["block_size"]] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(d, batches=20):
    model.eval()
    loss = sum(model(*get_batch(d))[1].item() for _ in range(batches)) / batches
    model.train()
    return loss


model = GPT(len(chars), **config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=args.iters, pct_start=0.05)
n_params = sum(p.numel() for p in model.parameters())
print(f"{len(text):,} chars, vocab {len(chars)}, {n_params / 1e6:.1f}M params, device {device}")

best_val, start = float("inf"), time.time()
for it in range(1, args.iters + 1):
    x, y = get_batch(train_data)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if it % 250 == 0 or it == args.iters:
        val = estimate_loss(val_data)
        print(f"iter {it:5d} | train {estimate_loss(train_data):.3f} | val {val:.3f} | {time.time() - start:.0f}s")
        if val < best_val:  # keep the checkpoint that generalizes best
            best_val = val
            torch.save({"model": model.state_dict(), "config": config, "chars": chars}, ckpt_path)

print(f"Done. Best val loss {best_val:.3f}, saved to {ckpt_path}")
