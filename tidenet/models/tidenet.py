import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Time2Vec, MLPHead, gather_last, masked_mean_std


class TIDENet(nn.Module):
    def __init__(self, d_x, d_c, d_s, hp, slope_init=0.0):
        super().__init__()
        H, d_t = hp["hidden"], hp["d_t"]
        self.H, self.hp = H, hp
        self.use_iiw = hp.get("use_iiw", True)
        self.use_proc = hp.get("use_process_view", True)
        self.hidden_decay = hp.get("hidden_decay", True)
        self.clip = tuple(hp.get("iiw_clip", (0.1, 10.0)))
        self.eps = hp.get("iiw_eps", 1e-6)
        self.w_min = hp.get("w_min", 1e-3)
        mode = hp.get("time_embedding", "t2v")

        self.in_w = nn.Parameter(torch.rand(d_x) * 0.1)
        self.in_b = nn.Parameter(torch.zeros(d_x))
        self.phi_t = Time2Vec(d_t, "t2v")
        self.phi_dt = Time2Vec(d_t, mode)
        self.phi_L = Time2Vec(d_t, "t2v")
        in_dim = 2 * d_x + self.phi_t.out_dim + self.phi_dt.out_dim + d_c

        self.layers = hp.get("layers", 2)
        self.cells = nn.ModuleList([nn.GRUCell(in_dim if l == 0 else H, H) for l in range(self.layers)])
        self.decay = nn.ModuleList([nn.Linear(self.phi_dt.out_dim, H) for _ in range(self.layers)])
        self.drop = nn.Dropout(hp.get("dropout", 0.2))

        self.v_lam = nn.Linear(H, 1)
        self.w_raw = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("grid", torch.linspace(0.0, hp.get("eed_grid_max_months", 120.0),
                                                    hp.get("eed_grid_points", 256)))

        self.W_a = nn.Linear(H, H, bias=False)
        self.U_a = nn.Linear(self.phi_L.out_dim, H, bias=False)
        self.q = nn.Linear(H, 1, bias=False)

        z_dim = 2 * H + (4 if self.use_proc else 0) + d_s
        self.head = MLPHead(z_dim, H, hp.get("dropout", 0.2), slope_init)

    @property
    def w_lam(self):
        return self.w_min + F.softplus(self.w_raw)

    def encode(self, b):
        x, m, last, delta = b["x"], b["m"], b["last"], b["delta"]
        gamma_x = torch.exp(-torch.relu(self.in_w * delta + self.in_b))
        x_tilde = m * x + (1 - m) * gamma_x * last
        e_t, e_dt = self.phi_t(b["t"]), self.phi_dt(b["dt"])
        v = torch.cat([x_tilde, m, e_t, e_dt, b["c"]], dim=-1)
        B, V, _ = v.shape
        h = [v.new_zeros(B, self.H) for _ in range(self.layers)]
        tops = []
        for j in range(V):
            inp = v[:, j]
            for l in range(self.layers):
                hl = h[l]
                if self.hidden_decay:
                    hl = hl * torch.exp(-torch.relu(self.decay[l](e_dt[:, j])))
                h[l] = self.cells[l](inp, hl)
                inp = self.drop(h[l])
            tops.append(h[-1])
        return torch.stack(tops, dim=1), e_dt

    def forward(self, b, proc_scale=1.0, return_aux=False):
        hs, e_dt = self.encode(b)
        vmask, length = b["vmask"], b["length"]
        B, V, _ = hs.shape
        w = self.w_lam
        c = self.v_lam(hs).squeeze(-1).clamp(max=8.0)
        b_lam = self.v_lam.bias.expand(B, 1)
        c_prev = torch.cat([b_lam, c[:, :-1]], dim=1)

        gap_next = torch.cat([b["dt"][:, 1:], b["dt"].new_zeros(B, 1)], dim=1)
        last_mask = F.one_hot(length - 1, V).bool() & vmask
        gap_next = torch.where(last_mask, b["gap"].unsqueeze(1).expand(B, V), gap_next)
        wd = (w * gap_next).clamp(max=12.0)
        integral = torch.exp(c) * torch.expm1(wd) / w
        ll_obs = c + wd - integral
        ll_cens = -integral
        ll = torch.where(last_mask, ll_cens, ll_obs) * vmask.float()
        tpp_ll = ll.sum(1) / length.float()

        log_lam = (c_prev + (w * b["dt"]).clamp(max=12.0))
        lam_hat = torch.exp(log_lam.clamp(max=12.0))
        mf = vmask.float()
        lam_bar = (lam_hat * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1.0)
        if self.use_iiw:
            omega = (lam_bar / lam_hat.clamp(min=self.eps)).clamp(*self.clip).detach()
        else:
            omega = torch.ones_like(lam_hat)

        score = self.q(torch.tanh(self.W_a(hs) + self.U_a(self.phi_L(b["lmt"])))).squeeze(-1)
        logits = (score + torch.log(omega)).masked_fill(~vmask, float("-inf"))
        attn = torch.softmax(logits, dim=1)
        z_traj = (attn.unsqueeze(-1) * hs).sum(1)

        h_last = gather_last(hs, length)
        if self.hidden_decay:
            h_last = h_last * torch.exp(-torch.relu(self.decay[-1](self.phi_dt(b["gap"]))))
        parts = [z_traj, h_last]

        z_proc = None
        if self.use_proc:
            c_N = gather_last(c.unsqueeze(-1), length).squeeze(-1)
            log_lam_L = c_N + w * b["gap"]
            u = self.grid.view(1, -1)
            surv = torch.exp(-(torch.exp(c_N).unsqueeze(1) * torch.expm1((w * u).clamp(max=12.0)) / w).clamp(max=50.0))
            e_next = torch.trapz(surv, self.grid, dim=1) / 12.0
            mean_ll, sd_ll = masked_mean_std(log_lam, vmask)
            z_proc = torch.stack([log_lam_L, e_next, mean_ll, sd_ll], dim=1)
            parts.append(z_proc * proc_scale)
        parts.append(b["s"])
        eta, mu, logvar = self.head(torch.cat(parts, dim=1))
        out = dict(eta=eta, mu=mu, logvar=logvar, tpp_ll=tpp_ll)
        if return_aux:
            out.update(attn=attn, omega=omega, lam_hat=lam_hat, c=c, w=w.detach(), z_proc=z_proc)
        return out
