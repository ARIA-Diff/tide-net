import torch
import torch.nn.functional as F


def total_loss(out, batch, soft_label=True, loss_reg="hetero", lambda_reg=0.3, lambda_tpp=0.1, slope_scale=5.0):
    target = batch["p"] if soft_label else batch["y"]
    cls = F.binary_cross_entropy_with_logits(out["eta"], target)
    parts = {"cls": cls.detach()}
    loss = cls
    if loss_reg != "none" and lambda_reg > 0:
        has = batch["has_beta"]
        if loss_reg == "hetero":
            var = out["logvar"].exp() + batch["se"] ** 2
            l = (batch["beta"] - out["mu"]) ** 2 / (2 * var) + 0.5 * torch.log(var)
        else:
            l = 0.5 * ((batch["beta"] - out["mu"]) / slope_scale) ** 2
        reg = (l * has).sum() / has.sum().clamp(min=1.0)
        loss = loss + lambda_reg * reg
        parts["reg"] = reg.detach()
    if out.get("tpp_ll") is not None and lambda_tpp > 0:
        tpp = -out["tpp_ll"].mean()
        loss = loss + lambda_tpp * tpp
        parts["tpp"] = tpp.detach()
    return loss, parts
