import numpy as np


def ckd_epi_2021(scr, age, female):
    scr = np.asarray(scr, dtype=float)
    age = np.asarray(age, dtype=float)
    female = np.asarray(female).astype(bool)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    r = scr / kappa
    egfr = 142.0 * np.minimum(r, 1.0) ** alpha * np.maximum(r, 1.0) ** -1.200 * 0.9938 ** age
    return egfr * np.where(female, 1.012, 1.0)


def creatinine_from_egfr(egfr, age, female):
    egfr = np.maximum(np.asarray(egfr, dtype=float), 1e-3)
    age = np.asarray(age, dtype=float)
    female = np.asarray(female).astype(bool)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    base = 142.0 * 0.9938 ** age * np.where(female, 1.012, 1.0)
    a = egfr / base
    r = np.where(a >= 1.0, a ** (1.0 / alpha), a ** (-1.0 / 1.2))
    return r * kappa
