import numpy as np
import os
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from dataset import make_concentric_rings
from model import ToyResNet2D

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, X, y, config):
    train_cfg = config["train"]
    set_seed(train_cfg["seed"])

    device = train_cfg["device"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["lr"]
    batch_size = train_cfg["batch_size"]

    model.to(device)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:02d} | loss {loss_sum/total:.4f} | acc {correct/total:.3f}")
    
def main():
    ROOT = Path(__file__).resolve().parent
    checkpoint_dir = ROOT / "checkpoints"
    config = {
        "model": {"depth": 8, "hidden": 64, "use_ln": False, "num_classes": 3},
        "train": {"seed": 0, "epochs": 30, "lr": 2e-3, "batch_size": 256, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    }

    X, y = make_concentric_rings()
    model = ToyResNet2D(**config["model"])
    train_model(model, X, y, config)

    model_name = checkpoint_dir / 'ln.pt' if config["model"]["use_ln"] else checkpoint_dir / 'no_ln.pt'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "config": config,
                }, model_name)

if __name__ == "__main__":
    main()