# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run A/B comparison (DAT-H3 vs baseline) - RECOMMENDED
python -m dat_ml.compare data/raw/your_data.csv --name test_comparison

# Run training on a dataset (DAT-H3 only)
python -m dat_ml.train data/raw/your_data.csv --name experiment_name --epochs 100

# Quick test import
python -c "from dat_ml import DAT_H3_Predictor, DEVICE; print(f'Using: {DEVICE}')"
```

## Architecture

DAT-ML applies principles from Discrete Alignment Theory and H3-Hybrid-Discovery to machine learning prediction.

### Core Concepts

- **E6 → H3 Projection**: Input data is embedded into 6D E6 lattice space, then projected to 3D H3 manifold (parallel + perpendicular components)
- **Phason Steering**: Learnable 4D rotations in perpendicular space for topological optimization
- **Icosahedral Network Blocks**: 12-way expansion reflecting H3 manifold symmetry
- **Topological Loss Functions**: Enforce stability principles (4.2x localization, -7.68ε energy depth)

### Module Structure

- `dat_ml/core/` - Device config (M3 Ultra optimized), experiment configuration
- `dat_ml/transforms/` - E6 projection, H3 hybrid compression transforms
- `dat_ml/models/` - H3-inspired neural network architecture
- `dat_ml/losses/` - Topological stability loss functions
- `dat_ml/data/` - Multi-format data loaders (time series, physical systems, networks, tabular)
- `dat_ml/evaluation/` - Blind test framework with benchmark comparisons

### Key Principles

1. **Blind Evaluation**: Test set is NEVER seen during training. Evaluated once at the end.
2. **A/B Comparison**: Always train a matched baseline (same parameter count) alongside DAT-H3
3. **Statistical Significance**: Use paired t-tests and Wilcoxon tests to validate improvement claims
4. **Benchmark Comparison**: Compare against analytical baselines (linear, persistence, moving average)

### A/B Comparison Framework

The `compare.py` module runs controlled experiments:
- Trains DAT-H3 model and baseline on **identical data splits**
- Baseline options: `matched` (same params), `mlp`, `transformer`
- Reports improvement with p-values and confidence intervals
- Outputs verdict on whether DAT-H3 framework provides advantage

### Hardware Target

Optimized for Apple M3 Ultra: MPS backend, 96GB unified memory, 20 performance + 8 efficiency cores.
