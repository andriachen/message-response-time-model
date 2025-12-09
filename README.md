# Bayesian Modeling of Message Response Times

A Bayesian statistical analysis comparing three probability distributions for modeling text message response time data using PyMC.

## Overview

This project explores how well different probability distributions capture real-world messaging behavior. Using iPhone text message data, I fit three models and compare them using Bayes Factors:

- **Exponential** — assumes constant response probability over time
- **Gamma** — allows for multi-stage response processes  
- **Weibull** — captures time-varying response probability

## Key Findings

| Model Comparison | Bayes Factor | Interpretation |
|------------------|--------------|----------------|
| Gamma vs Exponential | 88.6 | Strong evidence for Gamma |
| Weibull vs Exponential | 3,979 | Overwhelming evidence for Weibull |
| Weibull vs Gamma | 44.9 | Strong evidence for Weibull |

The Weibull distribution best captures messaging behavior: people either respond quickly or become progressively less likely to respond as time passes.

## Methods

- Prior selection using Mathematica's `Manipulate` for visualization
- Posterior sampling via PyMC's MCMC
- Model comparison using Sequential Monte Carlo (SMC) for marginal likelihood estimation
- Prior/posterior predictive checks for qualitative assessment

## Tech Stack

- Python
- PyMC
- ArviZ
- Matplotlib
- Pandas
- NumPy

## Files

- `analysis.py` — main analysis script
- `messages.csv` — collected response time data
- `discussion.pdf` — full write-up with discussion
- `*_plot_*.pdf` — generated visualizations

## Usage
```bash
pip install pymc arviz matplotlib pandas numpy
python analysis.py
```

## Course

CS134 - Computational Methods of Cognitive Science, Tufts University (Fall 2025)
