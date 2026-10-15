# ABCD_CausalML


Machine-learning–based causal interface for identifying and estimating effects of high-dimensional environmental predictors of substance use initiation in the ABCD Study.

Mengman Wei\*, Lasya Yadlapati, and Qian Peng\*

---

## Overview

This repository implements an **ML-based causal interface**, producing effect estimates with uncertainty.

The workflow is designed to help move from **high-dimensional longitudinal predictors** to **interpretable causal effect estimates** for modifiable risk factors.

> **Data note:** ABCD data are controlled access. This repo provides code only; it does not include ABCD data.


---

## Data access and responsible use

This repository provides analysis code and does not include participant-level ABCD Study data. Users must obtain authorized access through the relevant official data provider and comply with applicable data-use agreements, approved research purposes, and institutional requirements.

The software license does not grant access to ABCD data or permission to redistribute external datasets or restricted derived outputs.

Generated analysis files may contain participant identifiers, longitudinal measurements, predictors, and substance-use outcomes. Do not upload these files to public repositories or share them with unauthorized individuals.

For reproducibility, document the data release/version, accession or DOI where available, preprocessing steps, and exact code commit used. Include acknowledgments required by the data provider.

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

If you use or adapt this project, please cite:


*A Machine-Learning–Based Causal Interface for Environmental Predictors of Substance Use Initiation in the ABCD Study*  
Mengman Wei, Lasya Yadlapati, and Qian Peng

---

## Validation status and updates

The scripts currently posted may reflect an earlier development version. Their correspondence with the code used for the reported analyses has not yet been verified.

The repository will be reviewed against available later script versions and analysis records, and updated where needed. Issues identified during source-code review apply to the posted version; their relevance to later versions and their effects on the associated study results have not been established.

The review conducted to date included source-code inspection and limited synthetic checks. It did not include end-to-end execution or independent reproduction of the study results. Undetected errors or limitations may remain.

Updates will document the versions reviewed, corrections made, validation performed, and any known effects on outputs. Users should record the exact commit used and independently assess the code’s suitability for their intended analyses.
