from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/used_cars_sample.csv")
MODEL_PATH = Path("models/used_car_price_model.joblib")
FEATURES = ["make", "year", "mileage", "horsepower", "fuel_type", "body_type"]
TARGET = "price"


def build_model(regressor):
    categorical = ["make", "fuel_type", "body_type"]
    numeric = ["year", "mileage", "horsepower"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def train():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    candidates = {
        "ridge": build_model(Ridge(alpha=1.0)),
        "random_forest": build_model(RandomForestRegressor(n_estimators=250, random_state=42)),
        "gradient_boosting": build_model(GradientBoostingRegressor(random_state=42)),
    }
    results = []
    best_name = None
    best_score = float("-inf")
    best_model = None
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        results.append(
            {
                "model": name,
                "r2": round(score, 4),
                "mae": round(mean_absolute_error(y_test, preds), 2),
                "rmse": round(root_mean_squared_error(y_test, preds), 2),
            }
        )
        if score > best_score:
            best_name = name
            best_score = score
            best_model = model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "features": FEATURES, "best_model": best_name, "results": results}, MODEL_PATH)
    return pd.DataFrame(results)


if __name__ == "__main__":
    print(train().to_string(index=False))
