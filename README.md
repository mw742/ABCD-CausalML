# ABCD_CausalML


Machine-learning–based causal interface for identifying and estimating effects of high-dimensional environmental predictors of substance use initiation in the ABCD Study.

Mengman Wei\*, Lasya Yadlapati, and Qian Peng\*

---

## Overview

This repository implements an **ML-based causal interface**, producing effect estimates with uncertainty.

The workflow is designed to help move from **high-dimensional longitudinal predictors** to **interpretable causal effect estimates** for modifiable risk factors.

> **Data note:** ABCD data are controlled access. This repo provides code only; it does not include ABCD data.

---

## Repository Content

| File | Purpose |
|------|---------|
| `01_build_causal_panel_dataset.R` | Build the analysis dataset(s) used by downstream steps (feature engineering, inclusion/exclusion rules, outcomes, covariates/PRS, exports). |
| `02_discover_graph_granger.py` | Discover directed/temporal relationships among variables (graph discovery). |
| `03_estimate_effects_dml.py` | Estimate causal effects for selected exposures using DML / orthogonalized ML-based estimation. |



---

## Requirements

R/4.3.0 python/3.11.4 pytorch/2.5.0py311-cuda


---

## Citation:

If you use or adapt this project, please cite the associated manuscript:


*A Machine-Learning–Based Causal Interface for Environmental Predictors of Substance Use Initiation in the ABCD Study*  
Mengman Wei\*, Lasya Yadlapati, and Qian Peng\*
