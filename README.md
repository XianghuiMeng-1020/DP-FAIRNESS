# Excess Prioritization Turnover under Differential Privacy

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Opacus](https://img.shields.io/badge/Opacus-1.6-1B4F72)](https://opacus.ai/)
[![License](https://img.shields.io/badge/use-research-0F766E)](#scope)

A reproducible scientific package for studying **capacity-constrained human-review queues** when educational risk models are trained with **DP-SGD**.

This repository asks a narrow, operational question:

> When a model ranks students for a fixed-size review queue, does differentially private training change *who appears in the queue* **beyond ordinary non-private retraining**?

It does **not** assign support, treatments, or “deservingness.”

---

## Scientific object

At a prediction checkpoint $\tau$, a model produces scores $s_i$. A review queue $Q_k$ keeps the top $k$ fraction of records (default grid: 5%, 10%, 20%, 30%). Ties are broken by descending score, then lexicographic `record_id`.

Two non-private models on the **same split** give ordinary turnover $T_{\mathrm{NP}}$. A DP model, matched in split and initialization seed to the first non-private run, gives $T_{\mathrm{DP}}$. The central quantity is

$$
\mathrm{EPT}=T_{\mathrm{DP}}-T_{\mathrm{NP}}.
$$

Any of $\mathrm{EPT}>0$, $\mathrm{EPT}\approx 0$, or $\mathrm{EPT}<0$ is scientifically acceptable. The package also records ROC-AUC, Precision@$k$, and Outcome-Capture@$k$ (capture of the **recorded** adverse label, not of latent need).

| This package studies | This package does not study |
| --- | --- |
| Queue membership for human review | Direct support allocation |
| Turnover beyond ordinary retraining | Causal interventions |
| Predictive ranking at a temporal checkpoint | Fairness / harm adjudication |
| Formal DP accounting (Opacus) | Membership inference on these models |

---

## What’s in the box

```text
tlt/
  datasets/     OULAD Day-14/28/56 and UCI697 semester-1 / enrollment loaders
  metrics/      Priority queue, EPT, bootstrap CIs, cutoff localization
  trainer.py    Shared torch logistic / MLP-small; NP Adam; DP-SGD via Opacus
  run_grid.py   Precommitted 20-replicate grid (resume-safe)
  analyze.py    EPT, queue utility, and accounting surfaces from frozen scores
  protocol.py   Frozen protocol loader
configs/tlt/tlt_protocol.yaml
tests/tlt/
```

**Primary cells.** OULAD Day 28 and UCI697 Semester 1; logistic and one-hidden-layer MLP (64 units); target $\varepsilon \in \{1,5,10\}$.

**Temporal policy.**

- OULAD: unique-student split; VLE events with `date ≤ τ`; assessments due *and* submitted by `τ`; **assessment scores are excluded** (OULAD records submission day, not grade-release time).
- UCI697 Semester 1: enrollment fields + first-semester curricular fields only. Second-semester columns are rejected at load time.

---

## Setup

```bash
git clone https://github.com/XianghuiMeng-1020/DP-FAIRNESS.git
cd DP-FAIRNESS
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

GPU is recommended for the full DP grid (the development runs used CUDA). CPU will work for tests and small smokes.

Place public datasets under `data/raw/`:

```text
data/raw/oulad/          # official OULAD CSVs, including studentVle.csv
data/raw/uci697/data.csv # UCI “Predict Students' Dropout and Academic Success”
```

OULAD: [analyse.kmi.open.ac.uk/open_dataset](https://analyse.kmi.open.ac.uk/open_dataset)  
UCI 697: [doi:10.24432/C5MC89](https://archive.ics.uci.edu/dataset/697)

---

## Reproduce

```bash
# unit tests (queue algebra + temporal guards; no full data needed for most tests)
python -m pytest tests/tlt -q

# build temporally valid tables and freeze the protocol
python -m tlt.preprocess

# smoke: one replicate
python -m tlt.run_grid --smoke

# primary grid (OULAD Day 28 + UCI697 Semester 1)
python -m tlt.run_grid

# secondary checkpoints (OULAD Day 14/56, UCI697 enrollment)
python -m tlt.run_grid --secondary

# EPT / Precision@k / Outcome-Capture@k / cutoff localization
python -m tlt.analyze
```

Scores are written to `artifacts/predictions/<run_id>/` and are **not** git-tracked. Keep them if you need bit-identical downstream tables.

Precommitted seeds: split `20260820`; NP-A / DP `20000 + replicate`; NP-B `30000 + replicate`. Do not drop a replicate because a number is inconvenient.

---

## Protocol (v1.1)

| Knob | Frozen value |
| --- | --- |
| Replicates | 20 |
| Capacities | 5% / 10% / 20% / 30% |
| Architectures | logistic, MLP-64 |
| DP | DP-SGD, Poisson sampling, clip 1.0, 30 epochs, no early stop |
| Targets | $\varepsilon \in \{1,5,10\}$, $\delta = 1/n_{\mathrm{train}}$ |
| Inference | training replicate; 5000-resample percentile bootstrap |

Version 1.1 is a **temporal-validity correction**: unverifiable OULAD assessment *scores* were removed. Seeds, capacities, privacy grid, and architectures are unchanged.

---

## Status

This is the **scientific package** behind an IEEE Transactions on Learning Technologies manuscript in preparation. No manuscript source is included.

If you use the code, please cite the repository and, when available, the paper.

```bibtex
@software{dp_fairness_tlt,
  title   = {Excess Prioritization Turnover under Differential Privacy},
  author  = {Meng, Xianghui},
  year    = {2026},
  url     = {https://github.com/XianghuiMeng-1020/DP-FAIRNESS}
}
```

---

## License and data

Code in this branch is released for research reproduction. Underlying student datasets remain under their original terms; they are **not** redistributed here.
