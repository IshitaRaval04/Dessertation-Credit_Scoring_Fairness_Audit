# Credit Scoring Fairness Audit: Machine Learning and Bias Mitigation

**MSc Data Analytics dissertation project | De Montfort University, Dubai | 2026**

A reproducible Python-based investigation of **credit-default prediction, demographic fairness and bias-mitigation trade-offs** across three credit-risk datasets. The project compares conventional predictive metrics with group fairness measures to examine whether a model that predicts credit risk well also produces comparable outcomes across demographic groups.

> **Research project, not a production lending system.** Fairness metrics identify differences in model outcomes; they do not, by themselves, establish discrimination or legal compliance.

## Project overview

The notebook implements a complete research workflow:

1. Load and inspect three real-world credit-risk datasets.
2. Prepare dataset-specific features and standardise the target as **1 = default / high risk** and **0 = non-default / acceptable risk**.
3. Explore class balance, missing values and relationships in the data.
4. Build leakage-aware preprocessing and train classification models.
5. Evaluate predictive performance and fairness across **gender and age groups**.
6. Compare baseline predictions with **Reweighing** and **Equalized Odds post-processing**.
7. Analyse the predictive-performance/fairness trade-off and export comparison tables.

## Datasets

| Dataset | Purpose | Notes |
| --- | --- | --- |
| **German Credit** | Classify applicants by credit risk | 1,000 observations in the dissertation analysis |
| **Home Credit Default Risk** | Predict payment difficulty | Notebook's default `FAST_MODE=True` uses a reproducible, stratified **60,000-row sample** of the prepared Home Credit file |
| **Taiwan Default of Credit Card Clients** | Predict credit-card default | 30,000 observations in the dissertation analysis; loaded from a legacy `.xls` file |

The notebook **does not bundle the datasets**. Obtain the relevant datasets from their original providers or permitted sources and place them in a directory the notebook can search. The Home Credit input expected by this notebook is a prepared/combined CSV, not the complete collection of raw Kaggle relational tables.

Expected input filenames (the notebook accepts the listed alternatives):

```text
german_credit_data.csv
# or german_credit_data(3).csv

Home_Credit_Default_Risk.csv
# or Home_Credit_Default_Risk_Final_Combined.csv

default of credit card clients.xls
# or default of credit card clients(1).xls
```

File discovery searches recursively under `/kaggle/input`, `/mnt/data`, and the current working directory. For local execution, create a `data/` folder and place the files there.

## Tools and methods

**Language and libraries:** Python, pandas, NumPy, scikit-learn, XGBoost, Fairlearn, Matplotlib, and `xlrd` for the Taiwan `.xls` file.

**Models:** Logistic Regression, Random Forest, XGBoost.

**Preprocessing:** Duplicate inspection/removal, training-only numerical median imputation, explicit categorical missing-value handling, one-hot encoding, feature scaling, and stratified train/calibration/test splits using scikit-learn pipelines.

**Predictive evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices, and ROC curves.

**Fairness evaluation:** Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD), audited by gender and age group. Protected attributes are retained for auditing rather than supplied as baseline predictive features.

**Bias mitigation:**
- **Reweighing:** Adjusts training sample weights by protected-group/outcome combinations without generating synthetic observations.
- **Equalized Odds post-processing:** Uses Fairlearn's `ThresholdOptimizer` with a separate calibration subset to derive group-aware decision rules.

**Trade-off analysis:** Compares F1 with a summary fairness gap, `max(|DPD|, |EOD|)`, and applies a transparent rank-sum comparison. The notebook does **not** use ROC-AUC to evaluate the post-processed binary decisions.

## Repository structure

```text
.
├── README.md
└── Credit_Scoring_Fairness_Audit_FAST_Kaggle.ipynb
```

Generated CSV reports are written to `/kaggle/working` on Kaggle, or to the current directory otherwise. They are not required to be committed to the repository.

## How to run

1. Download or clone this repository.
2. Use **Kaggle Notebooks** (recommended for the original workflow) or a local Jupyter environment.
3. Add the three required datasets with the expected filenames above.
4. Install the dependencies if they are not already available:

   ```bash
   pip install numpy pandas matplotlib scikit-learn xgboost fairlearn xlrd notebook
   ```

5. Open `Credit_Scoring_Fairness_Audit_FAST_Kaggle.ipynb` and select **Run All**.
6. Review the generated plots, comparison tables and research-question summaries.

The notebook checks for and attempts to install `xlrd`, `fairlearn`, and `xgboost` if missing. Other common scientific Python dependencies must already be installed.

### Runtime settings

The environment cell defines:

```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
CALIBRATION_SIZE_WITHIN_TRAIN = 0.15
FAST_MODE = True
MAX_ROWS = {"German": None, "Home Credit": 60000, "Taiwan": None}
SHOW_EXTRA_PLOTS = False
```

Set `FAST_MODE=False` to use the available full prepared Home Credit input, subject to compute and memory limits. This does not automatically download or reconstruct the source dataset.

## Research findings

The accompanying MSc dissertation reports these **baseline ROC-AUC** results:

| Dataset | Highest baseline ROC-AUC model | ROC-AUC |
| --- | --- | ---: |
| German Credit | Random Forest | **0.797** |
| Home Credit | XGBoost | **0.755** |
| Taiwan Credit Card Default | XGBoost | **0.778** |

The study observed demographic differences in model outcomes. **Bias-mitigation methods did not consistently reduce fairness gaps**; their effects depended on the dataset, model and protected group. These findings illustrate why predictive performance and fairness should be evaluated together.

**Reproducibility note:** The values above are reported in the written dissertation. Re-running the notebook with different data files, library versions or runtime settings can produce different results. Use the notebook's exported tables as the authoritative outputs for a specific run.

## Generated outputs

The final notebook section exports nine CSV tables:

```text
01_baseline_performance.csv
02_baseline_fairness.csv
03_reweighing_performance.csv
04_reweighing_fairness.csv
05_equalized_odds_performance.csv
06_equalized_odds_fairness.csv
07_full_comparison.csv
08_before_after_changes.csv
09_cross_dataset_balance.csv
```

The notebook also generates diagnostic visualisations, including class distributions, confusion matrices, ROC curves and fairness/performance comparisons.

## Limitations

- The datasets differ in size, class balance, schema and population; results should not be interpreted as directly interchangeable.
- The default fast configuration samples Home Credit to reduce execution time.
- Protected-group fairness metrics are diagnostic measures, not proof of individual-level fairness or causality.
- Equalized Odds post-processing requires protected-group labels during calibration and evaluation, raising practical governance considerations.
- This is an academic evaluation workflow; it is not a deployed credit-decision application.

## Author

**Ishita Raval**  
MSc Data Analytics — De Montfort University, Dubai  
GitHub: [@IshitaRaval04](https://github.com/IshitaRaval04)

## Academic context

Developed as part of the 2026 MSc dissertation **“Credit Scoring Fairness Audit: Assessing Bias and Bias-Mitigation Techniques in Machine Learning Models.”**

Please cite the original dataset providers and any underlying methodological sources if reusing the analysis in academic work.
