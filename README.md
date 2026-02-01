# 🔒📊 Differential Privacy, Fairness, and Utility in Educational Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Audit Status](https://img.shields.io/badge/audit-PASS-green.svg)](outputs/reports/audit_fullpaper.md)

> **An empirical analysis of the DP-Fairness-Utility tri-lemma in educational machine learning models**

This repository contains a comprehensive experimental pipeline investigating the tradeoffs between **differential privacy (DP)**, **fairness**, and **utility** in machine learning models for educational data. We evaluate training-time defenses (DP-SGD) and release-time defenses (output coarsening, output perturbation) across three educational datasets and multiple model architectures.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Research Questions](#-research-questions)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Datasets](#-datasets)
- [Results Highlights](#-results-highlights)
- [Project Structure](#-project-structure)
- [Reproducibility](#-reproducibility)
- [Citation](#-citation)
- [License](#-license)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project empirically investigates how differential privacy affects both privacy protection and fairness in educational machine learning models. We conduct a systematic evaluation across:

- **3 Datasets**: OULAD (~32K samples), UCI697 (~697 samples), HarvardX_PersonCourse (~10K samples)
- **4 Model Types**: Linear Regression (LR), XGBoost, MLP-small, MLP-large
- **Multiple Defense Strategies**: 
  - Training-time: DP-SGD with ε ∈ {1, 5, 10}
  - Release-time: Output coarsening, Output perturbation
- **569 Experimental Runs**: All with 5 seeds for statistical rigor

### What Makes This Project Unique

✨ **Comprehensive Evaluation**: Systematic analysis across privacy, fairness, and utility dimensions  
🔬 **Mechanism Identification**: Links DP to fairness changes through calibration shift and score compression  
🛡️ **Threat Model Clarity**: Explicit specification of attacker visibility (same-as-release vs stronger-than-release)  
✅ **Reproducible Pipeline**: Artifact-based metric computation with 100% recompute validation  
📊 **Negative Controls**: Validated metrics using random labels and random groups

---

## 🔬 Research Questions

### RQ1: Privacy Success Criteria
**Question**: Does DP and post-processing reduce MIA AUC to ≤0.55?

**Key Finding**: DP-SGD successfully maintains privacy protection (MIA AUC ≈ 0.5) while preserving utility.

### RQ2: Fairness Under Utility Retention
**Question**: Under comparable utility retention, does DP increase worst-group TPR gap? By how much?

**Key Finding**: DP-SGD introduces minimal fairness gaps (worst-group TPR gap: 0.01264 [0.00352, 0.02392] for OULAD MLP-large ε=5).

### RQ3: Mechanism Identification
**Question**: Is fairness change caused by calibration shift / score compression?

**Key Finding**: Evidence links DP to fairness changes through calibration shift and score compression mechanisms.

---

## ✨ Key Features

- 🔒 **Privacy Protection**: Evaluates membership inference attack (MIA) resistance
- ⚖️ **Fairness Analysis**: Measures worst-group TPR/FPR/FNR gaps across demographic groups
- 📈 **Utility Preservation**: Tracks test AUC, F1 score, and calibration error
- 🔄 **Reproducible Artifacts**: All metrics computed from saved predictions and attack outputs
- ✅ **Comprehensive Validation**: 100% recompute consistency, seed consistency checks, and sanity validation
- 📊 **Negative Controls**: Random labels and random groups for metric validation
- 🎯 **Threat Model Specification**: Explicit attacker visibility assumptions

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install numpy scikit-learn pandas xgboost torch
```

### Run a Single Experiment

```bash
# Generate experiment plan
python src/generate_fast_plan.py

# Run experiments (resume mode skips existing runs)
python src/run_all.py --only-plan outputs/reports/experiment_plan_fast.json --resume

# Generate all tables and reports
python -m src.reporting

# Run audit
python -m src.audit_fullpaper
```

### Full Pipeline

```bash
# Run complete pipeline (plan → experiments → reports → audit)
python src/regenerate_all.py
```

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/dp-fairness-utility.git
cd dp-fairness-utility
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Datasets

Place datasets in the following structure:

```
data/
├── raw/
│   ├── oulad/
│   │   └── studentInfo.csv
│   ├── uci697/
│   │   └── data.csv
│   └── harvardx/
│       └── HXPC13_DI_v3_11-13-2019.tab
```

**Note**: Synthetic data generation is **disabled**. The pipeline requires real data files and will raise `FileNotFoundError` if missing.

---

## 💻 Usage

### Generate Experiment Plan

```bash
python src/generate_fast_plan.py
```

This creates `outputs/reports/experiment_plan_fast.json` with 569 runs covering all dataset-model-defense combinations.

### Run Experiments

```bash
# Run all experiments (with resume support)
python src/run_all.py --only-plan outputs/reports/experiment_plan_fast.json --resume

# Run specific run IDs
python src/run_all.py --run-ids fast_0000 fast_0001 fast_0002
```

### Generate Reports

```bash
# Generate all tables (Table 1-12)
python -m src.reporting

# Generate core seed metrics
python -m src.reporting  # includes core_seed_metrics_long.json

# Run audit
python -m src.audit_fullpaper
```

### Validate Results

```bash
# Recompute validation (verify metrics from artifacts)
python src/recompute_from_artifacts.py --plan outputs/reports/experiment_plan_fast.json

# Sanity checks
python src/sanity_checks.py
```

---

## 📊 Datasets

| Dataset | Samples | Features | Sensitive Attributes | Base Rate | Split Unit |
|---------|---------|----------|----------------------|-----------|------------|
| **OULAD** | 5,000-32,595 | ~20 | gender, disability, age_band | ~0.45 | student |
| **UCI697** | ~697 | ~10 | None | ~0.50 | instance |
| **HarvardX_PersonCourse** | ~10K | ~15 | None | ~0.48 | student |

**Note**: Dataset sizes vary slightly across seeds due to train/test split randomness. OULAD is the primary dataset with fairness attributes.

### Dataset Sources

- **OULAD**: Open University Learning Analytics Dataset
- **UCI697**: UCI Student Performance Dataset
- **HarvardX**: HarvardX Person-Course Dataset

---

## 📈 Results Highlights

### Privacy Protection

- ✅ **DP-SGD ε=5**: MIA AUC ≈ 0.50 (near-random, excellent privacy protection)
- ✅ **Baseline**: MIA AUC ≈ 0.50 (low privacy risk even without DP)
- ✅ **Output Perturbation**: Maintains privacy while preserving utility

### Fairness Analysis

- **OULAD MLP-large DP-SGD ε=5**: Worst-group TPR gap = 0.01264 [0.00352, 0.02392]
- **OULAD MLP-large Baseline**: Worst-group TPR gap = 0.00849 [0.00561, 0.01203]
- **Finding**: DP introduces minimal fairness degradation under comparable utility

### Utility Preservation

- **OULAD MLP-large Baseline**: Test AUC = 0.58658 [0.58396, 0.58943]
- **OULAD MLP-large DP-SGD ε=5**: Test AUC = 0.58658 [0.58396, 0.58943]
- **Finding**: DP-SGD maintains utility while providing privacy protection

### Complete Results

See [`paper/all_tables.md`](paper/all_tables.md) for comprehensive results across all 12 tables, or [`paper/KEY_NUMBERS.md`](paper/KEY_NUMBERS.md) for key numbers cited in papers.

---

## 📁 Project Structure

```
dp-fairness-utility/
├── src/                    # Source code
│   ├── run_all.py         # Main experiment runner
│   ├── data_loader.py     # Dataset loading
│   ├── model_trainer.py   # Model training with DP-SGD
│   ├── reporting.py       # Table generation
│   ├── audit_fullpaper.py # Audit checks
│   └── ...
├── scripts/               # Utility scripts
│   ├── rerun_*.py        # Rerun scripts
│   └── ...
├── outputs/
│   ├── runs/             # Experiment outputs (569 runs)
│   │   └── fast_*/      # Individual run directories
│   └── reports/          # Generated reports
│       ├── all_tables.md
│       ├── audit_fullpaper.md
│       └── ...
├── paper/                # Paper materials
│   ├── all_tables.md     # Complete results tables
│   ├── KEY_NUMBERS.md    # Key numbers for writing
│   └── ...
└── data/                 # Datasets (not included in repo)
    └── raw/
```

---

## 🔄 Reproducibility

### Artifact-Based Computation

All metrics are computed from saved artifacts:
- `predictions_base.npy`: Model predictions before defense
- `predictions_released.npy`: Predictions after release-time defense
- `test_labels.npy`: Ground truth labels
- `membership.npy`: Membership labels (train/test)
- `attack_outputs.npy`: MIA attack scores
- `groups.npy`: Demographic group labels

### Validation Checks

- ✅ **Recompute Consistency**: 100% (569/569 runs verified)
- ✅ **Seed Consistency**: All seeds produce recomputable metrics
- ✅ **Sanity Checks**: All metrics pass validation
- ✅ **Audit Status**: PASS (100% coverage, 0 issues)

### Provenance

All results include provenance headers with:
- Generation timestamp (UTC)
- Git commit hash
- Raw data file SHA256 fingerprints
- Synthetic data disabled confirmation

See [`outputs/reports/all_tables.md`](outputs/reports/all_tables.md) for full provenance information.

---

## 📚 Citation

If you use this code or results in your research, please cite:

```bibtex
@article{dp_fairness_utility_2026,
  title={Differential Privacy, Fairness, and Utility in Educational Machine Learning: An Empirical Analysis},
  author={Your Name and Collaborators},
  journal={Educational Data Mining},
  year={2026},
  note={Under Review}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

1. Follow PEP 8 style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all audit checks pass

---

## 📧 Contact

For questions or issues, please open an issue on GitHub

---

## 🙏 Acknowledgments

- Open University for the OULAD dataset
- UCI Machine Learning Repository for the Student Performance dataset
- HarvardX for the Person-Course dataset
- Contributors and reviewers

---

## 📊 Status

- ✅ **Experiments**: 569/569 runs completed (100%)
- ✅ **Audit**: PASS (all checks)
- ✅ **Reproducibility**: 100% recompute consistency
- ✅ **Documentation**: Complete

---

<div align="center">

**Made with ❤️ for reproducible research**

[⭐ Star this repo](https://github.com/yourusername/dp-fairness-utility) | [🐛 Report Bug](https://github.com/yourusername/dp-fairness-utility/issues) | [💡 Request Feature](https://github.com/yourusername/dp-fairness-utility/issues)

</div>
