# 🏠 IMMO-ELIZA-ML
<p align="left">
<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg" />
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
<img src="https://img.shields.io/badge/🏠%20Immo--Eliza-Price%20Prediction-orange?style=flat-square" />
</p>




## 🚀 Project Overview

A machine‑learning pipeline that predicts Belgian real‑estate prices using structured property listing data.
Three regression models — Linear Regression, Random Forest, and XGBoost — are trained, cross‑validated, and evaluated.
The best-performing model is saved and used by predict.py to generate price predictions for new properties.




## 📁 Repo Structure
```
immo-eliza-ml/
├── data/               # raw and processed datasets             
├── models/             # saved trained pipeline (best_model.joblib)
├── reports/            # supporting analysis / write-ups
├── src/
│   ├── train.py        # builds, trains, evaluates, and saves the pipeline
│   └── predict.py      # loads the saved pipeline and predicts on new data
├── requirements.txt
└── README.md
```



## 🎯Approach

**1.🧽 Data Cleaning**

**clean_data() in train.py:**

The dataset is cleaned using the following steps:
- Remove duplicate listings (property_id or property_url)
- Drop irrelevant or high‑cardinality columns
(postal_code, address, city, property_url, etc.)

Logical corrections
- If has_garage = 0 → parking_count = 0
- If has_garden = 0 → garden_area_m2 = 0

Missing values are not imputed here — they are handled inside the ML pipeline to avoid leakage

**2. 🧹Preprocessing — Imputation, Scaling, Encoding**

Implemented using ColumnTransformer:

- Numerical features
  - Median imputation

  - Standard scaling

- Categorical features
  - Constant imputation (“Unknown”)

  - One‑hot encoding (ignore unseen categories)

All preprocessing is embedded inside the model pipeline → safe for production.

**3.🤖Model Comparison**

Three candidate models were evaluated:

       - Linear Regression
       - Random Forest Regressor
       - XGBoost Regressor

Each model is wrapped in a pipeline and evaluated consistently.



Hyperparameters were tuned manually and deliberately, rather than via
automated search (e.g. RandomizedSearchCV), so that every choice could be
explained and reasoned through directly.

**4. 📊 Cross-Validation**

Each pipeline is evaluated using 5‑fold cross‑validation:

- Provides a mean R² across 5 train/validation splits

- Confirms that performance is stable and not dependent on a single train/test split

- Helps detect overfitting or unstable models

**5. 💾Model Selection & Saving**

The model with the highest test R² is selected as the best pipeline and
saved to models/best_model.joblib via joblib.
predict.py loads this same file to generate predictions on new property data, with no manual
preprocessing required on the caller's side — the pipeline handles
imputation, scaling, and encoding internally.



## 🏆Results

| Model | Train R² | Test R² | CV R² (5-fold) | Test MAE (EUR) | Test RMSE (EUR) | Overfit Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.826 | 0.647 | 0.624 | 97,746 | 224,662 | 0.179 |
| **Random Forest** | 0.657 | 0.568 | 0.580 | 108,594 | 248,637 | 0.089 |
| **Linear Regression** | 0.494 | 0.497 | 0.440 | 131,929 | 268,282 | -0.0024 |

 **XGBoost was selected as the final model** because:
 - Test R² ≈ 0.65, matching expected benchmarks for this dataset
 - CV R² (0.624) is close to Test R² (0.647) → stable generalization
 - Overfit gap (0.179) is moderate and expected for boosting models
 - Regularization (reg_alpha, reg_lambda) and subsampling helped reduce overfitting

## 🛠️ Engineering Highlights & Design Decisions
**Zero Data Leakage:** Missing values are intentionally *not* imputed during the initial `clean_data()` phase. Instead, all statistical imputations are encapsulated strictly within the `ColumnTransformer` pipeline. This guarantees that test set metrics are completely untainted by training set distributions, mirroring true production ML behavior.

**Deliberate Hyperparameter Tuning:** Rather than relying on an automated `RandomizedSearchCV`, hyperparameters for the `RandomForest` and `XGBoost` models were tuned manually and progressively. This constraint was chosen deliberately to deeply analyze how individual parameters (such as `max_depth`, `min_child_weight`, and L1/L2 regularization coefficients) impact variance, bias, and the train/test generalization gap.

**Imputation Strategy Note**
Group‑based imputation (e.g., median by province or property_type) was explored but not used due to complexity and leakage risks.
The simpler global‑median approach inside SimpleImputer performed well and kept the pipeline clean and safe.



## 💻 Usage

### Prerequisites
It is highly recommended to run this project inside an isolated virtual environment to prevent package version conflicts (especially with `xgboost` and `scikit-learn`).


**Activate your data science environment (e.g., .venv )**

```bash
# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train and save the best model
```bash
python src/train.py
```
### Generate predictions on example property data
```bash
python src/predict.py
```
### Example output:
```bash
Property 1: predicted price = 506,229.34 EUR
Property 2: predicted price = 156,793.30 EUR
```

Both scripts resolve paths relative to their own location, so they can be
run from any working directory.



## 👩‍💻 Author

**Neha** — Machine Learning Engineer
Immo Eliza ML Project
