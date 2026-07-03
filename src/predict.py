"""
predict.py — loads the trained pipeline from train.py and predicts price
for new property data.
"""

import os
import pandas as pd
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")


def load_model(model_path: str = MODEL_PATH):
    """Loads the saved preprocessing+model pipeline."""
    return joblib.load(model_path)


def predict_price(pipeline, new_data: pd.DataFrame) -> pd.Series:
    """Takes raw new property data (same columns the model was trained on)
    and returns predicted prices. The pipeline handles imputation, scaling,
    and encoding internally -- no manual preprocessing needed here."""
    return pipeline.predict(new_data)


def main():
    pipeline = load_model()

    # dummy example properties, matching the FEATURES used in train.py
    dummy_data = pd.DataFrame([
        {
            "bedrooms": 3, "bathrooms": 1, "living_area_m2": 150,
            "total_area_m2": 300, "facades": 2, "has_garage": 1,
            "has_garden": 1, "latitude": 50.85, "longitude": 4.35,
            "property_type": "House", "province": "Brussels Capital Region",
            "state_of_the_building": "Good", "epc_score": "C",
            "has_elevator": 0, "parking_count": 1, "garden_area_m2": 100,
            "is_nearby_city_prestigious": 1,"kitchen_equipped":"Unknown",
            "building_year":1990,
        },
        {
            "bedrooms": 1, "bathrooms": 1, "living_area_m2": 55,
            "total_area_m2": None, "facades": 2, "has_garage": 0,
            "has_garden": 0, "latitude": 50.60, "longitude": 5.57,
            "property_type": "Apartment", "province": "Liège",
            "state_of_the_building": "To renovate", "epc_score": "F",
            "has_elevator": 1, "parking_count": 0, "garden_area_m2": 0,
            "is_nearby_city_prestigious": 0,"kitchen_equipped":"Super equipped",
            "building_year":1990,
        },
    ])

    predictions = predict_price(pipeline, dummy_data)

    for i, price in enumerate(predictions):
        print(f"Property {i + 1}: predicted price = {price:,.2f} EUR")


if __name__ == "__main__":
    main()