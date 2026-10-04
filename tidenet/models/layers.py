import torch
import torch.nn as nn


class Time2Vec(nn.Module):

    def __init__(self, dim: int, mode: str = "t2v"):
        super().__init__()
        self.mode = mode
        if mode == "raw":
            self.out_dim = 1
        else:
            self.out_dim = dim
            self.lin = nn.Linear(1, 1)
            self.sin = nn.Linear(1, dim - 1)

    def forward(self, t):
        t = t.unsqueeze(-1)
        if self.mode == "raw":
            return t
        return torch.cat([self.lin(t), torch.sin(self.sin(t))], dim=-1)


class MLPHead(nn.Module):

    def __init__(self, in_dim, hidden, dropout, slope_init=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.0, slope_init, 0.0]))

    def forward(self, z):
        o = self.net(z)
        return o[:, 0], o[:, 1], o[:, 2].clamp(-6.0, 6.0)


def gather_last(h, length):
    idx = (length - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
    return h.gather(1, idx).squeeze(1)


def masked_mean_std(v, mask, eps=1e-6):
    mf = mask.float()
    n = mf.sum(1).clamp(min=1.0)
    mean = (v * mf).sum(1) / n
    var = (((v - mean.unsqueeze(1)) ** 2) * mf).sum(1) / n
    return mean, (var + eps).sqrt()
