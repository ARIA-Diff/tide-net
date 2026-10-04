import math

import torch
import torch.nn as nn

from ..data import constants as C
from .layers import Time2Vec, MLPHead, gather_last


def _filled(b):
    return b["m"] * b["x"] + (1 - b["m"]) * b["last"]


class SeqModel(nn.Module):

    def __init__(self, encoder, z_dim, d_s, hidden, dropout, slope_init=0.0):
        super().__init__()
        self.encoder = encoder
        self.head = MLPHead(z_dim + d_s, hidden, dropout, slope_init)

    def forward(self, b, **_):
        z = self.encoder(b)
        eta, mu, logvar = self.head(torch.cat([z, b["s"]], dim=1))
        return dict(eta=eta, mu=mu, logvar=logvar, tpp_ll=None)


class GRUD(nn.Module):

    def __init__(self, d_x, d_c, H):
        super().__init__()
        self.H = H
        self.in_w = nn.Parameter(torch.rand(d_x) * 0.1)
        self.in_b = nn.Parameter(torch.zeros(d_x))
        self.dec_h = nn.Linear(d_x, H)
        self.cell = nn.GRUCell(2 * d_x + d_c, H)
        self.out_dim = H

    def forward(self, b):
        x, m, last, delta, c = b["x"], b["m"], b["last"], b["delta"], b["c"]
        gx = torch.exp(-torch.relu(self.in_w * delta + self.in_b))
        xh = m * x + (1 - m) * gx * last
        B, V, _ = x.shape
        h = x.new_zeros(B, self.H)
        hs = []
        for j in range(V):
            h = h * torch.exp(-torch.relu(self.dec_h(delta[:, j])))
            h = self.cell(torch.cat([xh[:, j], m[:, j], c[:, j]], dim=-1), h)
            hs.append(h)
        return gather_last(torch.stack(hs, 1), b["length"])


class TLSTM(nn.Module):

    def __init__(self, d_x, d_c, H):
        super().__init__()
        self.H = H
        self.Wd = nn.Linear(H, H)
        self.W = nn.Linear(d_x + d_c + H, 4 * H)
        self.out_dim = H

    def forward(self, b):
        xf = torch.cat([_filled(b), b["c"]], dim=-1)
        B, V, _ = xf.shape
        h = xf.new_zeros(B, self.H)
        c = xf.new_zeros(B, self.H)
        hs = []
        for j in range(V):
            cs = torch.tanh(self.Wd(c))
            g = 1.0 / torch.log(math.e + b["dt"][:, j]).unsqueeze(-1)
            c = c - cs + cs * g
            i, f, o, u = self.W(torch.cat([xf[:, j], h], dim=-1)).chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u)
            h = torch.sigmoid(o) * torch.tanh(c)
            hs.append(h)
        return gather_last(torch.stack(hs, 1), b["length"])


class ODERNN(nn.Module):

    def __init__(self, d_x, d_c, H, steps=2):
        super().__init__()
        self.H, self.steps = H, steps
        self.f = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, H))
        self.cell = nn.GRUCell(2 * d_x + d_c, H)
        self.out_dim = H

    def _evolve(self, h, dt):
        dt = (dt / 12.0).unsqueeze(-1) / self.steps
        for _ in range(self.steps):
            k1 = self.f(h)
            k2 = self.f(h + 0.5 * dt * k1)
            k3 = self.f(h + 0.5 * dt * k2)
            k4 = self.f(h + dt * k3)
            h = h + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return h

    def forward(self, b):
        xf = _filled(b)
        B, V, _ = xf.shape
        h = xf.new_zeros(B, self.H)
        hs = []
        for j in range(V):
            h = self._evolve(h, b["dt"][:, j])
            h = self.cell(torch.cat([xf[:, j], b["m"][:, j], b["c"][:, j]], dim=-1), h)
            hs.append(h)
        return gather_last(torch.stack(hs, 1), b["length"])


class MTAN(nn.Module):

    def __init__(self, d_x, d_c, H, d_t, heads, n_ref, dropout, win_years=5.0):
        super().__init__()
        self.t2v = Time2Vec(d_t)
        self.q_proj = nn.Linear(d_t, H)
        self.k_proj = nn.Linear(d_t, H)
        self.v_proj = nn.Linear(2 * d_x + d_c, H)
        self.attn = nn.MultiheadAttention(H, heads, dropout=dropout, batch_first=True)
        self.gru = nn.GRU(H, H, batch_first=True)
        self.register_buffer("ref", torch.linspace(0.0, win_years, n_ref))
        self.out_dim = H

    def forward(self, b):
        B = b["x"].size(0)
        q = self.q_proj(self.t2v(self.ref.unsqueeze(0).expand(B, -1)))
        k = self.k_proj(self.t2v(b["t"]))
        v = self.v_proj(torch.cat([_filled(b), b["m"], b["c"]], dim=-1))
        out, _ = self.attn(q, k, v, key_padding_mask=~b["vmask"], need_weights=False)
        _, hN = self.gru(out)
        return hN[-1]


class STraTS(nn.Module):

    def __init__(self, d_x, d_c, H, layers, heads, dropout, max_triplets):
        super().__init__()
        self.nv = len(C.MEAS_IDX) + len(C.EVENT_IDX)
        self.K = max_triplets
        self.var_emb = nn.Embedding(self.nv, H)
        self.val_ff = nn.Sequential(nn.Linear(1, H), nn.Tanh(), nn.Linear(H, H))
        self.time_ff = nn.Sequential(nn.Linear(1, H), nn.Tanh(), nn.Linear(H, H))
        layer = nn.TransformerEncoderLayer(H, heads, 2 * H, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pool_w = nn.Linear(H, H)
        self.pool_q = nn.Linear(H, 1, bias=False)
        self.demo = nn.Sequential(nn.Linear(len(C.COMORB_IDX) + d_c, H), nn.Tanh())
        self.register_buffer("vid", torch.arange(self.nv))
        self.out_dim = 2 * H

    def forward(self, b):
        x, m, t, vm = b["x"], b["m"], b["t"], b["vmask"]
        B, V, _ = x.shape
        nv = self.nv
        val = x[:, :, :nv].reshape(B, V * nv)
        msk = ((m[:, :, :nv] > 0) & vm.unsqueeze(-1)).reshape(B, V * nv)
        tt = t.unsqueeze(-1).expand(B, V, nv).reshape(B, V * nv)
        vid = self.vid.view(1, 1, nv).expand(B, V, nv).reshape(B, V * nv)
        K = min(self.K, V * nv)
        top = torch.where(msk, tt, torch.full_like(tt, -1.0)).topk(K, dim=1).indices
        val, tt, vid, msk = (a.gather(1, top) for a in (val, tt, vid, msk))
        e = self.var_emb(vid) + self.val_ff(val.unsqueeze(-1)) + self.time_ff(tt.unsqueeze(-1))
        h = self.enc(e, src_key_padding_mask=~msk)
        a = self.pool_q(torch.tanh(self.pool_w(h))).squeeze(-1).masked_fill(~msk, float("-inf"))
        z = (torch.softmax(a, 1).unsqueeze(-1) * h).sum(1)
        stat = torch.cat([b["x"][:, :, C.COMORB_IDX], b["c"]], dim=-1)
        return torch.cat([z, self.demo(gather_last(stat, b["length"]))], dim=1)


class MonthlyLSTM(nn.Module):

    def __init__(self, d_x, d_c, H, layers, dropout, n_bins=60):
        super().__init__()
        self.n_bins = n_bins
        self.rnn = nn.LSTM(2 * d_x, H, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.out_dim = H

    @torch.no_grad()
    def _bin(self, b):
        x, m, vm = b["x"], b["m"] * b["vmask"].unsqueeze(-1), b["vmask"]
        B, V, D = x.shape
        idx = (self.n_bins - 1 - torch.floor(b["lmt"] * 12.0).clamp(0, self.n_bins - 1)).long()
        vals = x.new_zeros(B, self.n_bins, D)
        msk = x.new_zeros(B, self.n_bins, D)
        ar = torch.arange(B, device=x.device)
        for j in range(V):
            mj = m[:, j]
            bj = idx[:, j]
            cur = vals[ar, bj]
            vals[ar, bj] = torch.where(mj > 0, x[:, j], cur)
            msk[ar, bj] = torch.maximum(msk[ar, bj], mj)
        filled = []
        last = x.new_zeros(B, D)
        for k in range(self.n_bins):
            last = torch.where(msk[:, k] > 0, vals[:, k], last)
            filled.append(last)
        return torch.stack(filled, 1), msk

    def forward(self, b):
        vals, msk = self._bin(b)
        out, _ = self.rnn(torch.cat([vals, msk], dim=-1))
        return out[:, -1]


DEEP_BASELINES = ("grud", "tlstm", "mtan", "odernn", "strats", "lstm_monthly")


def build_baseline(name, hp, slope_init=0.0):
    H, d_x, d_c, d_s = hp["hidden"], C.D_X, C.D_C, 2
    drop = hp.get("dropout", 0.2)
    if name == "grud":
        enc = GRUD(d_x, d_c, H)
    elif name == "tlstm":
        enc = TLSTM(d_x, d_c, H)
    elif name == "odernn":
        enc = ODERNN(d_x, d_c, H, hp.get("ode_steps", 2))
    elif name == "mtan":
        enc = MTAN(d_x, d_c, H, hp["d_t"], hp.get("heads", 4), hp.get("n_ref_points", 32), drop)
    elif name == "strats":
        enc = STraTS(d_x, d_c, H, hp.get("strats_layers", 2), hp.get("heads", 4), drop,
                     hp.get("strats_max_triplets", 256))
    elif name == "lstm_monthly":
        enc = MonthlyLSTM(d_x, d_c, H, hp.get("layers", 2), drop)
    else:
        raise ValueError(name)
    return SeqModel(enc, enc.out_dim, d_s, H, drop, slope_init)
