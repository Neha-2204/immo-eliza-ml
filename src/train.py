"""
train.py — Immo Eliza ML: trains and saves the price-prediction pipeline.

Pipeline: clean_data -> select features -> preprocess (impute/scale/encode)
          -> train/test split -> train model -> evaluate -> save.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import os

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "properties_final_irene.csv")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")

# Columns dropped and reasoning 
COLS_TO_DROP = {
    "property_id":"Unique identifier,not required for prediction",
    "coord_swapped": "Not required, was done to get the info",
    "region": "Not needed as does not affect the price",
    "price_per_m2":"We are considering target as price and living area is already present",
    "property_url": "unique",
    "postal_code":"large number of unique values",
    "floors_total":"Not required as wont affect much",
    "address": "Won't affect the price much",
    "city":"Not considering city, latitude and longiture will work for now",

}   

FEATURES = [
    "bedrooms", "bathrooms", "living_area_m2", "total_area_m2", "facades",
    "has_garage", "has_garden", "latitude", "longitude",
    "property_type", "province", "state_of_the_building", "epc_score",
    "has_elevator", "parking_count", "garden_area_m2", "is_nearby_city_prestigious",
    "kitchen_equipped","building_year"
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw dataset by 
    - Removing duplicates
    - Dropping irrelaevant columns
    """
    df = df.copy()
    # Remove duplicate listings based on property_id or property_url
    id_col = "property_id" if "property_id" in df.columns else "property_url"
    df = df.drop_duplicates(subset=id_col)

   
   # 2. Drop irrelevant columns
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)

    # 3. Simple zero-filling rules--> If no garage,parking count must be 0
    if "parking_count" in df.columns and "has_garage" in df.columns:
        df.loc[df["has_garage"] == 0, "parking_count"] = 0

      # If no garden → garden_area_m2 must be 0
    if "garden_area_m2" in df.columns and "has_garden" in df.columns:
        df.loc[df["has_garden"] == 0, "garden_area_m2"] = 0

    # 4.  median filling for numeric columns
    #numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    #for col in numeric_cols:
        #df[col] = df[col].fillna(df[col].median())

    # 5.Categorical columns filled with 'Unknown'
    #categorical_cols = df.select_dtypes(include=["object"]).columns
    #for col in categorical_cols:
        #df[col] = df[col].fillna("Unknown")

    #print(df_clean.columns.tolist())
    print(f"Remaining shape: {df.shape}")
        
    return df


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Reusable preprocessing: median-impute+scale numerics,
    constant-impute+one-hot categoricals (unseen categories -> ignored)."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),   #imputate the median for the numerical values
        ('scaler', StandardScaler())                     # Standardize the just filled missing values
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def create_model_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Combines the preprocessor to any given regression model."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def evaluate_and_diagnose(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    """Computes key metrics and flags overfitting via train/test R2 gap."""
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    gap = train_r2 - test_r2

    print(f"\n--- {name} ---")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2:  {test_r2:.4f}")
    print(f"Test MAE: {test_mae:,.2f} EUR")
    print(f"Test RMSE: {test_rmse:,.2f} EUR")
    print(f"Overfit gap (Train R2 - Test R2): {gap:.4f}" + (" <- possible overfitting" if gap > 0.3 else ""))

    return {"name": name, "train_r2": train_r2, "test_r2": test_r2,
            "test_mae": test_mae, "test_rmse": test_rmse, "gap": gap}

def cross_validate_model(pipeline, X_train, y_train, cv_splits=5):
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
    print(f"Cross‑val R2 scores: {scores}")
    print(f"Mean CV R2: {scores.mean():.4f}")
    return scores.mean()


def main():
    df = pd.read_csv(DATA_PATH)
    df_clean = clean_data(df)

    X = df_clean[FEATURES]
    y = df_clean["price"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            max_depth=10,
            min_samples_leaf=7,
            max_features=0.7,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            max_depth=6,   # for better genralization
            n_estimators=300, #more trees
            learning_rate=0.03, #slower learning, less overfitting
            min_child_weight=3, #requires more data per leaf
            subsample =0.9,  #each tree sees 90% of rows
            colsample_bytree=0.9,     # NEW — each tree sees 90% of features
            reg_alpha=0.05,            # NEW — L1 regularization
            reg_lambda=1.0,           # NEW — L2 regularization
            random_state=RANDOM_STATE
        ),
    }

    

    results = []
    fitted_pipelines = {}
    for name, model in candidates.items():
        print(f"\n=== {name} ===")
        pipeline = create_model_pipeline(preprocessor, model)
        cv_r2 = cross_validate_model(pipeline, X_train, y_train, cv_splits=5) #cross validation before fitting
        
        pipeline.fit(X_train, y_train)
       
        metrics = evaluate_and_diagnose(name, pipeline, X_train, X_test, y_train, y_test) #evaluation on test metrics
        metrics["cv_r2"] = cv_r2
        
        fitted_pipelines[name] = pipeline
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
    print("\n=== Model comparison (sorted by Test R2) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["name"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH)
    print(f"Saved best pipeline to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()