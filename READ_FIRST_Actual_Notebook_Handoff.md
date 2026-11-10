# CKD-NHANES Research — Verified Teammate Handoff

**Source of truth:** uploaded `Untitled8.ipynb` (58 notebook cells: 29 code, 29 markdown). This guide preserves the order and original step labels found in the notebook. It is **not** a reconstruction from conversation history. Individual original source-code cells and their recorded plain-text outputs are saved in the companion folders. Figures and HTML tables are preserved in the original `.ipynb`, which must be shared with this handoff.

**Research title:** *A Machine Learning-Based Multiclass Classification of Chronic Kidney Disease Stages: Feature Selection and Model Optimization.*

## A. Exact notebook sequence and what each cell did

| Notebook cell | Actual notebook heading / code step | What happened |
|---:|---|---|
| 1 | Universal NHANES EDA | Uploaded and explored data; adds `source_file` in the combined overview (30 columns), distinct from actual 29-column original CSV. |
| 3–4 | EDA and CKD class distribution | Re-uploaded the original 11,933 × 29 CSV and examined 8 labels including Unknown. |
| 6 | **STEP 1.2 — COMPLETE 7-CLASS CKD LABEL VALIDATION** | Audited GFR stage thresholds, ACR distributions, label patterns; exported label audit. |
| 8 | **STEP 3 — CLEAN TARGET & STRATIFIED TRAIN/TEST SPLIT** | Excluded Unknown labels, split into 5,060 training and 1,266 test records; initially 24 predictors. |
| 10 | **STEP 4 — 3-FOLD STRATIFIED CROSS-VALIDATION SETUP** | Created 3 training/validation folds, kept test separate. |
| 13 | **STEP 5A — DETAILED MISSING-VALUE INVESTIGATION** | Examined missingness over 24 predictors on training data; no imputation here. |
| 15 | **STEP 5B — MISSING VALUE HANDLING (CORRECTED)** | Excluded three heavily missing predictors; imputed 21 predictors (16 numeric, 5 categorical), fold-safe. |
| 17 | **STEP 6 — CATEGORICAL ENCODING & NUMERICAL SCALING** | Demonstrated standard scaling / one-hot encoding, 21→39 columns in example fold; prepared SMOTENC inputs. |
| 19 | **STEP 7 — CORRELATION ANALYSIS** | Pearson and Spearman relationships between numeric predictors; did not drop correlated features. |
| 22 | **STEP 8A — FEATURE IMPORTANCE AND RANKING** | Mutual Information + Random Forest ranking using first training fold; did not train predictive experiment here. |
| 24 | **STEP 8B — FEATURE SET CREATION (UPDATED)** | Created F1, F2, F3, F4; F2 selected dynamically inside training folds in subsequent experiments. |
| 26 | **STEP 9A — SMOTENC CLASS IMBALANCE HANDLING (UPDATED)** | Studied and demonstrated fold-specific SMOTENC and class handling. |
| 29 | **STEP 10A — BASELINE MODEL COMPARISON (UPDATED F1-F4)** | Initial XGBoost and Random Forest results across feature sets/sampling variants. |
| 32 | **STEP 11A — XGBOOST HYPERPARAMETER TUNING** | CV HPT of XGBoost. |
| 34 | **STEP 11B — RANDOM FOREST HYPERPARAMETER TUNING** | CV HPT of Random Forest. |
| 36 | **STEP 11C — EXTRA TREES BASELINE + HPT** | Baseline and tuned Extra Trees comparisons. |
| 38 | **STEP 11D — LIGHTGBM & CATBOOST BASELINE COMPARISON** | Baselines of LightGBM and CatBoost. |
| 40 | **STEP 11E — SVM + LOGISTIC REGRESSION BASELINES** | SVM and Logistic Regression baselines. |
| 43 | **STEP 11F — LIGHTGBM + CATBOOST HPT** | CV HPT of LightGBM and CatBoost. |
| 45 | **STEP 11G — COMPLETE MISSING F2 COMPARISONS** | Filled additional F2 baseline coverage. |
| 47 | **STEP 12 — CONSOLIDATED MODEL COMPARISON** | Aggregated earlier tables; reported missing baseline combinations. |
| 48 | **STEP 12B — FIXED COMPLETE BASELINE COMPARISON** | Fixed coverage: 56/56 required baseline combinations, plus 4 additional results. |
| 50 | **STEP 14A — SIX-MODEL STACKING (HPT-AWARE)** | Six base learners and LR meta, 3-fold outer+inner CV across F1–F4. **There is no Step 13A three-model stacking cell in this uploaded notebook.** |
| 52 | **STEP 14 — FINAL STACKING TRAIN/TEST EVALUATION** | Evaluated selected F3 six-model ensemble on the test set. |
| 53 | **STEP 14B — STANDALONE MODEL TEST COMPARISON** | Compared 7 standalone models on all 4 sets using the test data. |
| 55 | **STEP 15A — CKD HIGH-ACCURACY AUDIT** | Audited exact duplicate rows and train/test overlap, feature-set sizes. |
| 56 | **STEP 14B — FINAL STACKING TEST FOR ALL FOUR FEATURE SETS** | Evaluated the six-model stack on F1–F4 on test data. **Step 14B is reused as a label in two different notebook cells.** |
| 57 | **STEP 15 — SHAP ANALYSIS FOR F1, F2, F3, F4** | LightGBM SHAP, trained on training data; 150 sampled training records per set. |

> There is no separate notebook cell labeled Step 2, and the recorded step identifiers are not strictly sequential. Do **not** renumber these steps when documenting the code.

## B. Dataset and label validation — actual saved outputs

The original file in cell 3 had **11,933 rows and 29 columns**, with **zero duplicate full records**. Cell 1's **30 columns** arose from adding a `source_file` column during combined-file EDA; this does not mean the original dataset has 30 variables.

| Label | Number of records |
|---|---:|
| No CKD | 3,592 |
| Stage 1 (Kidney Damage) | 382 |
| Stage 2 (Mildly Decreased) | 1,875 |
| Stage 3a (Mild-Moderate) | 341 |
| Stage 3b (Moderate-Severe) | 97 |
| Stage 4 (Severely Decreased) | 28 |
| Stage 5 (Kidney Failure) | 11 |
| Unknown | 5,607 |

Cell 6 recorded **100% agreement with expected eGFR ranges** for the seven named stage groups, but this **does not mean clinical CKD diagnoses were validated**. In Stage 2, its ACR table showed **1,645 A1 (<30), 179 A2, 20 A3 and 31 missing**. The No CKD group was predominantly A1; Stage 1 was A2/A3. ACR and single-visit eGFR cannot establish chronicity or other unobserved kidney-damage evidence; conclusions must remain about *dataset-defined stage labels*.

Cell 8 split the **6,326 known-label rows** into **5,060 train and 1,266 test** (24 predictors initially). This notebook uses the full known-label population; do **not** say this train/test sample is adults-only without a separate age filtering step. Three stratified training folds have ~3,373 training and ~1,687 validation rows.

## C. Preprocessing, feature ranking, F1–F4

Cells 13–17: the missingness study initially inspected 24 predictors (16 numerical, 8 categorical). Cell 15 removed `insulin_use`, `diabetes_pills`, and `current_smoker`, leaving **21 features: 16 numerical and 5 categorical**. Numeric imputation and categorical missing-category treatment were fitted on training folds. Cell 17's one-hot example yielded **39 encoded columns**, but this varies by fold and feature set. SMOTENC was applied on training folds only in the later modeling pipeline; the held-out validation and test rows were not oversampled.

Cell 19 examined Pearson and Spearman feature correlations without removing features. Cell 22 performed Mutual Information + Random Forest feature ranking on a training fold. Cell 24 constructed:

- **F1 (21):** `age`, `gender`, `ethnicity`, `education_level`, `poverty_income_ratio`, `bmi`, `weight_kg`, `height_cm`, `bp_systolic`, `bp_diastolic`, `serum_creatinine`, `blood_urea_nitrogen`, `albumin_serum`, `phosphorus`, `bicarbonate`, `calcium`, `uric_acid`, `urine_creatinine`, `urine_albumin`, `diabetes_diagnosed`, `ever_smoked`.
- **F2 (15):** data-driven MI+RF top-ranked variables; **chosen on each training fold**. In cell 57 the full-training SHAP F2 selection was: `serum_creatinine`, `age`, `blood_urea_nitrogen`, `urine_albumin`, `uric_acid`, `bp_systolic`, `education_level`, `phosphorus`, `bmi`, `albumin_serum`, `weight_kg`, `poverty_income_ratio`, `height_cm`, `ever_smoked`, `urine_creatinine`. Do not substitute an earlier chat list; fold-specific selections can differ.
- **F3 (15):** `serum_creatinine`, `age`, `urine_albumin`, `urine_creatinine`, `blood_urea_nitrogen`, `uric_acid`, `bp_systolic`, `bp_diastolic`, `diabetes_diagnosed`, `bmi`, `phosphorus`, `bicarbonate`, `albumin_serum`, `calcium`, `gender`.
- **F4 (16):** F1 excluding `serum_creatinine`, `age`, `gender`, `urine_albumin`, and `urine_creatinine`. F4 retains other blood-test predictors; it is not a lab-free early screening model.

## D. Baselines, tuning, and consolidated comparison

Cells 29–48 systematically evaluated **seven standalone models**: XGBoost, Random Forest, Extra Trees, LightGBM, CatBoost, SVM, and Logistic Regression. Cross-validation also contrasted with and without training-only SMOTENC. HPT was performed for the five tree ensembles XGBoost, Random Forest, Extra Trees, LightGBM, and CatBoost. Hyperparameter selection did **not** always improve performance relative to baseline.

**Important correction from notebook:** Cell 47 (Step 12) flagged missing baseline configurations, so the initial master table was **incomplete**. Cell 48 (Step 12B) is the corrected master comparison: **56 of 56 required combinations**, plus **4 preserved additional results**. Use **Step 12B** rather than Step 12 for the complete final baseline coverage. The detailed fold results, tuning grids, winning parameter values, and full accuracy/F1 tables are preserved in the corresponding cell-output text files and notebook itself; do not infer parameters from an isolated summary line.

## E. Stacking — exact notebook steps

Cell 50 is titled **STEP 14A — SIX-MODEL STACKING (HPT-AWARE)** and evaluates **XGBoost, LightGBM, Random Forest, Extra Trees, CatBoost and SVM** as six base learners. Logistic Regression is the **meta-learner**. Six 7-class probability vectors produce **42 meta-features**. With **outer 3-fold** performance evaluation and **inner 3-fold** out-of-fold probability generation, both preprocessing and dynamic F2 selection stay within training subsets. The uploaded notebook **does not include a separate 3-model stacking implementation**; do not claim it was executed in this exact notebook.

**Step 14A (development cross-validation, not held-out test)**

| Feature set | CV accuracy | CV macro-F1 | CV balanced accuracy | CV macro-F1 SD |
|---|---:|---:|---:|---:|
| F1 | 96.15% | 86.68% | 85.91% | 5.20 pp |
| F2 | 87.96% | 75.62% | 72.71% | 6.34 pp |
| F3 | 96.19% | 86.91% | 85.96% | 3.34 pp |
| F4 | 64.96% | 28.34% | 26.59% | 3.91 pp |

F3 was selected by CV on the development training set, narrowly outperforming F1 in mean macro-F1. The difference is small relative to fold variation; do not claim statistical significance.

## F. Held-out test — exact saved results

Cell 52 initially evaluated six-base stacking **only on F3**, using 3 training-only OOF folds, fitting six final bases on full 5,060 training rows, then evaluating 1,266 test rows: **accuracy 98.03%, macro-F1 93.33%, weighted-F1 98.01%, balanced accuracy 92.71%; 1,241 correct / 25 wrong**. Test support: No CKD 719, Stage 1 77, Stage 2 375, Stage 3a 68, Stage 3b 19, Stage 4 6, Stage 5 2. The rare Stage 4/5 estimates are unstable.

Cell 53 then evaluated all 7 standalone algorithms across all 4 sets on held-out test. Standalone **LightGBM F3: accuracy 97.63%, macro-F1 94.42%**; **LightGBM F1: accuracy 97.39%, macro-F1 93.67%**. Thus stacking was not best on every metric. Consult `Cell_53_Recorded_Output.txt` for the complete 28-row standalone results.

Cell 56 reran the six-base stack for F1–F4 on the held-out test. These are the **actual reported values** (not cross-validation averages):

| Feature set | Test accuracy | Test macro-F1 | Test balanced accuracy |
|---|---:|---:|---:|
| F1 | 97.87% | **94.18%** | 93.23% |
| F2 | 92.10% | 87.87% | 87.36% |
| F3 | **98.03%** | 93.33% | 92.71% |
| F4 | 66.43% | 39.49% | 35.44% |

**Important finding:** F3 had the highest **stacking test accuracy**, while **F1 had the highest stacking test macro-F1**. F3 remained the initially CV-selected final configuration. The broader test comparisons followed examination of F3 test outcomes; describe them as **exploratory**, not fresh confirmatory model selection.

Cell 55 audited exact feature-row duplication: **0 duplicates in train, 0 in test, 0 exact cross-split matches, 0 conflicting labels for identical feature rows**. This does not eliminate target-definition dependence or unobserved patient-level duplication.

## G. SHAP — Step 15 in notebook cell 57

LightGBM standalone was fitted for each feature set, and SHAP was computed on **150 sampled training records** per set. These SHAP values explain **LightGBM**, not the entire six-model stack. Top five features from the saved output:

| F1 | F2 | F3 | F4 |
|---|---|---|---|
| serum_creatinine | serum_creatinine | serum_creatinine | blood_urea_nitrogen |
| age | age | age | uric_acid |
| urine_albumin | urine_albumin | urine_albumin | bp_systolic |
| gender_Female | height_cm | gender_Female | education_level_Missing |
| urine_creatinine | urine_creatinine | urine_creatinine | albumin_serum |

SHAP reveals model reliance, not causality. High F1–F3 performance with target-defining inputs and lower F4 performance strongly suggest *target-definition dependence*; they do not prove that the model can diagnose CKD earlier than standard kidney tests. The feature sets also differ in more than a single variable, so no single individual feature's effect is isolated by F3 versus F4.

## H. Thesis interpretation and evidence boundaries

**Defensible research objective:** compare feature-selection strategies, imbalance-aware evaluation, tuning and ensemble methods for **seven-class dataset-defined CKD stage labels**, and examine dependence on the variables underlying staging calculations. The high scores may represent learning the label-construction rules, not superior clinical diagnosis. Stage 1/2 label validity, repeated-measurement chronicity, tiny advanced-stage sample sizes, no external validation, and the reused test set require explicit discussion.

**Do not claim**: a validated early-detection system without laboratory tests; a novel diagnostic formula; that 98.03% is clinical accuracy; that all base learners were tuned; that SHAP explains stacking; that this notebook contains a 3-model stack; or that Step 12 rather than Step 12B has complete baseline coverage.

## I. How teammate should use the package

1. Read this report first, then the `Cell_Index.csv`.
2. Read each `Cell_##.py` and matching `Cell_##_Recorded_Output.txt` in actual notebook order.
3. Open the **original attached `Untitled8.ipynb`**, which preserves figures/HTML outputs not reproducible in plain text.
4. Build thesis methodology from the code, and results tables from the corresponding saved output—not from guessed histories.
5. Compare with verified peer-reviewed literature; verify dataset authorship, staging definitions, and external citations separately.

