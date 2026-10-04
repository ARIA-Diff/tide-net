import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from .data.dataset import collate, epoch_batches, sequential_batches
from .losses import total_loss


def to_device(b, device):
    return {k: v.to(device) for k, v in b.items()}


def make_loader(ds, batches, num_workers=0):
    return DataLoader(ds, batch_sampler=batches, collate_fn=collate, num_workers=num_workers)


@torch.no_grad()
def predict(model, ds, device, batch_size=256, num_workers=0):
    model.eval()
    probs, mus, sigmas = [], [], []
    for b in make_loader(ds, sequential_batches(len(ds), batch_size), num_workers):
        o = model(to_device(b, device))
        probs.append(torch.sigmoid(o["eta"]).cpu())
        mus.append(o["mu"].cpu())
        sigmas.append(torch.exp(0.5 * o["logvar"]).cpu())
    return pd.DataFrame({"prob": torch.cat(probs).numpy(), "mu": torch.cat(mus).numpy(),
                         "sigma": torch.cat(sigmas).numpy()})


def fit(model, ds_train, ds_val, loss_kw, ocfg, device, seed, num_workers=0, log=print):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ocfg["lr"], weight_decay=ocfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ocfg["max_epochs"]) if ocfg.get("cosine_decay") else None
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(ocfg["max_epochs"]):
        model.train()
        batches = epoch_batches(ds_train.inst, ocfg["landmarks_per_patient"], ocfg["batch_size"], rng)
        tot, nb = 0.0, 0
        for b in make_loader(ds_train, batches, num_workers):
            b = to_device(b, device)
            loss, _ = total_loss(model(b), b, **loss_kw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["grad_clip"])
            opt.step()
            tot, nb = tot + float(loss), nb + 1
        if sched:
            sched.step()
        pv = predict(model, ds_val, device, num_workers=num_workers)["prob"].values
        auprc = float(average_precision_score(ds_val.y > 0.5, pv))
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_auprc=auprc))
        log(f"epoch {ep:03d}  loss {tot / max(nb, 1):.4f}  val AUPRC {auprc:.4f}")
        if auprc > best:
            best, best_state, bad = auprc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= ocfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), best
