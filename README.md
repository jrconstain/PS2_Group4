# PS2 — Group 4 · Predicting Household Poverty in Colombia

**Course:** Big Data & Machine Learning for Applied Economics (2025‑2)  
**Professor:** Ignacio Sarmiento Barbieri
**Goal:** Build ML models to **predict monetary poverty at the household level** in Colombia, maximizing **F1** with a **parsimonious** feature set.


**Best public‑LB result:** **XGBoost**, F1 ≈ **0.74** (threshold calibrated with out‑of‑fold predictions).
🏆 **Public Leaderboard:** Ranked **#1** — see the [Kaggle competition leaderboard](https://www.kaggle.com/competitions/uniandes-bdml-2025-20-ps-2/leaderboard).

**Project paper:** see the `document/` folder (PDF + sources).

---

## 1) Repository structure

```
PS2_Group4/
├─ document/     # Final paper (PDF/TeX) and appendices
├─ scripts/      # Reproducible R Markdown scripts for data & models
├─ stores/       # Raw/processed data, model artifacts, metrics, submissions
└─ views/        # Final figures & tables used in the paper/slides
```

- **document/**: final write‑up and appendices (compiled from `views/`).  
- **scripts/**: self‑contained Rmds to clean data, engineer features, and train models (Logit/Elastic Net, CART, Random Forest, GBM, XGBoost, preliminary LightGBM).  
- **stores/**: input datasets, intermediate stores, metrics (`*_metrics.csv`), model objects (`*.rds`), and Kaggle submissions.  
- **views/**: all figures/tables exported by the scripts and referenced by the paper.

This organization follows the instructor’s template: clear documentation, readable code, and reproducibility.

---

## 2) Data (overview)

- **Source:** GEIH 2018 (DANE) with official poverty lines.  
- **Unit:** households and individuals; target is `poor = 1{income_pc < poverty_line}`.  
- **Splits:** a **train** partition with income (to define the label) and a **test** partition **without income** to avoid trivial solutions and to emulate real‑world deployment.

Large files are handled with Git LFS or referenced externally when exceeding GitHub limits.

---

## 3) Results (brief)

- **Best model:** **Extreme Gradient Boosting (XGBoost)** with 5‑fold CV and **threshold calibration for F1** using out‑of‑fold predictions: **F1 ≈ 0.74**, with strong precision/recall trade‑off.  
- **Key features:** household **labor intensity** and **human capital** (e.g., share/avg of employed, hours, education including household head), **household structure**, and **poverty line context**.  
- **Comparators:** Random Forest (F1 ≈ 0.71) and Gradient Boosting (F1 ≈ 0.72) trail the final XGBoost.  
- **Class imbalance:** threshold tuning (PR‑curve) and class weights tested for robustness.

See the paper in `document/` for full details (hyperparameters, tuning strategy, variable importance, and model comparison tables).

---

## 4) Reproducibility

### 4.1 Requirements

- **R ≥ 4.3**  
- Main packages: `tidyverse`, `data.table`, `janitor`, `caret`, `glmnet`, `rpart`, `ranger`, `gbm`, `xgboost`, `pROC`, `PRROC`, `rmarkdown`, `knitr`, `kableExtra`.

> We recommend using `p_load` from `pacman`.

```r
# Quick install with pak
install.packages("pacman")
p_load("tidyverse","data.table","janitor","caret","glmnet","rpart",
           "ranger","gbm","xgboost","pROC","PRROC","rmarkdown","knitr","kableExtra")
```

### 4.2 Pipeline (execution order)

1. **Data prep**
   - `scripts/data_manipulation.Rmd` → cleaning, merges, and household‑level feature engineering.
   - `scripts/data_final.Rmd` → imputations and final stores for modeling.

2. **Modeling**
   - `scripts/M1_logit models.Rmd` → baselines (Logit 1–3) + threshold tuning.
   - `scripts/M2_Trees models.Rmd` → CART and Random Forest (incl. VI‑based pruning tests).
   - `scripts/M3_BOOSTING_MODELS.Rmd` → Gradient Boosting (Bernoulli) with OOF thresholding.
   - `scripts/M0_XGBoosting.Rmd` → final XGBoost (grid, CV=5, OOF threshold).
   - `scripts/M4_LightGBM.Rmd` → preliminary LightGBM exploration.

---

## 5) Deliverables & grading (course summary)

- **PDF paper** (≤ 12 pages, ≤ 10 exhibits; free appendix).  
- **Slides**: three decks (`data_team_##`, `othermodels_team_##`, `best_team_##`).  
- **GitHub repo** with a useful README, replication instructions, and ≥ 5 substantial commits per member on `main`.  
- **Kaggle submissions**: at least 8, using ≥ 5 distinct algorithms from the syllabus.  
- **AER‑style tables**, clean and well‑commented code, and reproducible results.

---

## 6) Re‑creating a Kaggle submission

1. Run the pipeline up to the chosen model (e.g., `M0_XGBoosting.Rmd`).  
2. Verify the prediction CSV in `stores/` (column names/format per competition spec).  
3. Upload to the course Kaggle competition and record the reported **F1**.

---

## 7) Method notes

- **Threshold optimization:** maximize F1 using **OOF predictions** and PR‑curve‑based search.  
- **Imputation rules:** tailored per variable type; corrected edge cases identified during EDA.  
- **Parsimony vs. performance:** VI‑based reduction was tested; full engineered set retained for best F1.

---

## 8) Authors

- Juan José Rojas · Francisco Soler · Jesús Yancy  
Contact: please open an issue in this repository.

---

## 9) License

Shared for academic purposes. Check DANE/GEIH data usage terms before redistribution.

---

### Best Model (one‑slide summary)
**XGBoost (binary, CV=5, OOF‑thresholded), F1 ≈ 0.74.**  
**What worked:** labor‑intensity & human‑capital features + thorough tuning under class imbalance.
