# TIDE-Net

Code for *Modeling informative visits and label uncertainty to predict rapid kidney function decline from irregularly sampled electronic health records*.

TIDE-Net combines a time-aware GRU, a neural temporal point process for the visit intensity, inverse-intensity weighted attention pooling, and soft-label training on the sampling distribution of the outcome-window eGFR slope.

## Requirements

Python 3.10+ and the packages in `requirements.txt` (PyTorch 2.3, LightGBM, statsmodels, DuckDB).

```bash
pip install -r requirements.txt
```

## Data

MIMIC-IV v3.1 requires credentialed access through [PhysioNet](https://physionet.org/content/mimiciv/). Place the `hosp/` and `icu/` folders under `data/mimic-iv-3.1/` or set `paths.mimic_dir` in `configs/data.yaml`. No patient data are included in this repository.

## Usage

```bash
python scripts/build_cohort.py
python scripts/search_hparams.py
python scripts/run_models.py --models reference deep deep_sl tidenet
python scripts/run_models.py --models tidenet --outcome slope3
python scripts/run_models.py --models tidenet --split temporal
python scripts/run_models.py --ablations a_no_tpp_loss b_no_iiw c_no_process_view d_hard_labels e_no_heteroscedastic f_no_hidden_decay g_raw_delta
python scripts/run_semisynthetic.py
python scripts/aggregate_results.py --against strats lightgbm
python scripts/explain_cases.py --run-dir outputs/runs/random_slope5/tidenet/seed0 --ig
```

All settings are in `configs/`; any entry can be changed with `--override key=value`. Results are written to `outputs/`.

## Layout

```
configs/    cohort, training, search space, ablations, semi-synthetic and evaluation settings
scripts/    command-line entry points
tidenet/
  data/       cohort extraction, landmarks, outcomes and soft labels, splits, batching
  models/     TIDE-Net and deep comparators
  baselines/  KFRE, LMM extrapolation, logistic regression, LightGBM
  losses.py, train.py, evaluate.py, semisynthetic.py, interpret.py, search.py, experiment.py
```

## Notes

- MIMIC-IV item identifiers are listed in `tidenet/data/constants.py` and should be checked against the `d_labitems` dictionary of the release in use.
- The weight of the low-eGFR term in the semi-synthetic visit process is set in `configs/semisynthetic.yaml`.

## License

Code is released under the MIT License. MIMIC-IV is subject to the PhysioNet credentialed data use agreement.
