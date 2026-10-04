import pandas as pd
import torch

from .data import constants as C
from .data.dataset import LandmarkDataset, collate
from .train import to_device


@torch.no_grad()
def explain_instance(model, ds: LandmarkDataset, i: int, device, grid_months=None):
    model.eval()
    b = to_device(collate([ds[i]]), device)
    o = model(b, return_aux=True)
    n = int(b["length"][0])
    row = ds.inst.iloc[i]
    visits = pd.DataFrame({
        "t_years_from_window_start": b["t"][0, :n].cpu().numpy(),
        "egfr_standardised": b["x"][0, :n, C.EGFR_COL].cpu().numpy(),
        "egfr_measured": b["m"][0, :n, C.EGFR_COL].cpu().numpy(),
        "setting": b["c"][0, :n].argmax(-1).cpu().numpy(),
        "attention": o["attn"][0, :n].cpu().numpy(),
        "iiw_weight": o["omega"][0, :n].cpu().numpy(),
        "lambda_hat": o["lam_hat"][0, :n].cpu().numpy(),
    })
    sigma = float(torch.exp(0.5 * o["logvar"][0]))
    summary = dict(subject_id=int(row["subject_id"]), L=float(row["L"]), prob=float(torch.sigmoid(o["eta"][0])),
                   slope=float(o["mu"][0]), slope_lo=float(o["mu"][0]) - 1.96 * sigma,
                   slope_hi=float(o["mu"][0]) + 1.96 * sigma)
    curves = None
    if grid_months is not None:
        g = torch.as_tensor(grid_months, dtype=torch.float32, device=device)
        curves = torch.exp(o["c"][0, :n].unsqueeze(1) + o["w"] * g.unsqueeze(0)).cpu().numpy()
    return summary, visits, curves


def integrated_gradients(model, ds: LandmarkDataset, device, n_steps=50, batch_size=64, max_instances=2000):
    model.eval()
    n = min(len(ds), max_instances)
    tot_x = torch.zeros(C.D_X, device=device)
    tot_p = 0.0
    alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=device)
    for s in range(0, n, batch_size):
        b = to_device(collate([ds[i] for i in range(s, min(s + batch_size, n))]), device)
        B = b["x"].size(0)
        gx, gl, gp = torch.zeros_like(b["x"]), torch.zeros_like(b["x"]), torch.zeros(B, 1, device=device)
        for a in alphas:
            bb = dict(b)
            bb["x"] = (b["x"] * a).requires_grad_(True)
            bb["last"] = (b["last"] * a).requires_grad_(True)
            ps = torch.full((B, 1), float(a), device=device, requires_grad=True)
            f = torch.sigmoid(model(bb, proc_scale=ps)["eta"]).sum()
            inputs = [bb["x"], bb["last"]] + ([ps] if model.use_proc else [])
            grads = torch.autograd.grad(f, inputs)
            gx += grads[0] / n_steps
            gl += grads[1] / n_steps
            if model.use_proc:
                gp += grads[2] / n_steps
        attr = (b["x"] * gx + b["last"] * gl).sum(1).abs()
        tot_x += attr.sum(0).detach()
        tot_p += float(gp.abs().sum())
    ser = pd.Series((tot_x / n).cpu().numpy(), index=C.X_VARS)
    proc = tot_p / n
    share = proc / max(float(ser.sum()) + proc, 1e-12)
    ser = ser.sort_values(ascending=False)
    ser["visit_process"], ser["process_share"] = proc, share
    return ser
