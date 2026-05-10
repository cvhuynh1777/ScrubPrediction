# Launch Scrub Prediction from Weather

A probabilistic machine learning model that predicts the likelihood of a rocket launch being scrubbed due to weather — designed as a complement to NASA's deterministic Lightning Launch Commit Criteria (LLCC).

**GitHub:** [cvhuynh1777/ScrubPrediction](https://github.com/cvhuynh1777/ScrubPrediction)

---

## Overview

NASA's LLCC defines binary go/no-go thresholds for launch safety. This project asks a different question: given current atmospheric conditions, what is the *probability* of a scrub? A continuous risk score gives mission control early warning before any single rule trips.

---

## Repository Contents

```
launch_scrub_study/
├── launch_scrub_prediction.ipynb   # Full analysis notebook
├── merged_launch_weather.csv       # Merged launch + weather dataset (116 records)
├── README.md                       # This file
└── data/                           # Raw ERA5 weather CSVs (not included — see below)
```

### Note on raw data

The `data/` folder contains ERA5 hourly reanalysis CSVs for four U.S. launch sites (Cape Canaveral, KSC, Vandenberg, Boca Chica) spanning ~80 years. These files total several GB and are excluded from the repo.

To reproduce the full pipeline from scratch:

1. Create an account at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
2. Download ERA5 hourly data (`reanalysis-era5-single-levels`) for the site coordinates and variables listed in Section 2 of the notebook
3. Place CSVs in `data/` matching the filenames referenced in cell `ec193442`

The merged output (`merged_launch_weather.csv`) is included so the modeling sections (Sections 3–7) can be run without re-downloading ERA5.

---

## Data Sources

| Source | Description |
|--------|-------------|
| `all_space_missions.csv` | NASA/SpaceX launch records with mission outcomes |
| ERA5 (ECMWF) | Hourly reanalysis weather at launch site coordinates |

---

## Methodology

- **Split first, then scale** — stratified 80/20 train/test split before any transformation
- **No oversampling** — class imbalance handled via `class_weight='balanced'` inside each model (avoids leakage from pre-split SMOTE)
- **Year excluded** — temporal confounding: older launches fail more due to technology, not weather
- **5-fold stratified CV** — all metrics reported with mean ± std across folds

---

## Results Summary

| Model | CV F1 | CV ROC-AUC |
|---|---|---|
| Logistic Regression | 0.251 ±0.134 | 0.632 ±0.199 |
| **Random Forest** *(selected)* | 0.133 ±0.267 | 0.610 ±0.120 |
| Gradient Boosting | 0.000 ±0.000 | 0.657 ±0.084 |
| SVM (RBF) | 0.252 ±0.159 | 0.561 ±0.071 |

Random Forest was selected for interpretability — feature importances map directly to LLCC physical rules. AUC 0.56–0.66 is above random chance but reflects the primary constraint: only ~13 scrub events across 116 records.

---

## Key EDA Finding

Scrubbed launches cluster under **warmer, lower-pressure, higher-humidity** conditions — directionally consistent with LLCC physics and the most statistically reliable takeaway from this dataset.

---

## Environment

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Python 3.11 · scikit-learn 1.3+

---

## Future Work

- Expand with CCAFS/FAA dedicated scrub records
- Add electric field and lightning density as direct LLCC proxies
- Incorporate 6-hour rolling weather trends
- Calibrate output probabilities with Platt scaling

---

*Christina Huynh · M.S. Computational Data Analytics, Georgia Tech*
