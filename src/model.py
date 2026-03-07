import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualFFN2D(nn.Module):
    def __init__(self, hidden=64, use_ln=True):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.ln = nn.LayerNorm(hidden) if use_ln else nn.Identity()
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, h):
        z = self.fc1(h)          
        z = self.ln(z)           
        z = F.gelu(z)
        r = self.fc2(z)          
        return h + r, r


class ToyResNet2D(nn.Module):
    def __init__(self, depth=6, hidden=64, use_ln=True, num_classes=3):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualFFN2D(hidden=hidden, use_ln=use_ln) for _ in range(depth)])
        self.head = nn.Linear(2, num_classes)

    def forward(self, x, return_traj=False):
        h = x
        hs = [h]
        rs = []
        for blk in self.blocks:
            h, r = blk(h)
            hs.append(h)
            rs.append(r)
        logits = self.head(h)
        if return_traj:
            return logits, hs, rs
        return logits