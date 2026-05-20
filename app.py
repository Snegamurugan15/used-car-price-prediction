from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from train_model import DATA_PATH, FEATURES, MODEL_PATH, train


app = Flask(__name__)


def load_bundle():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)


PAGE = """
<!doctype html>
<html>
<head>
  <title>DrivenInsights Used Car Pricing</title>
  <style>
    body {font-family: Segoe UI, sans-serif; margin: 36px; background: #f7f9fb; color: #17202a;}
    main {max-width: 920px; margin: auto;}
    form, section {background: white; padding: 22px; border: 1px solid #dce3ea; border-radius: 8px; margin-bottom: 18px;}
    label {display:block; margin-top: 12px; font-weight: 600;}
    input, select {width: 100%; padding: 10px; border: 1px solid #c9d3df; border-radius: 6px;}
    button {margin-top: 18px; padding: 11px 18px; border: 0; border-radius: 6px; background: #155e75; color: white; font-weight: 700;}
    .price {font-size: 2.4rem; color: #155e75; font-weight: 800;}
  </style>
</head>
<body>
<main>
  <h1>DrivenInsights Used Car Price Prediction</h1>
  <p>Flask inference app backed by a scikit-learn training pipeline.</p>
  <form method="post">
    <label>Make</label><select name="make">{make_options}</select>
    <label>Year</label><input name="year" type="number" value="2021">
    <label>Mileage</label><input name="mileage" type="number" value="45000">
    <label>Horsepower</label><input name="horsepower" type="number" value="210">
    <label>Fuel Type</label><select name="fuel_type">{fuel_options}</select>
    <label>Body Type</label><select name="body_type">{body_options}</select>
    <button>Predict price</button>
  </form>
  {prediction}
  <section><h2>API</h2><code>POST /predict</code> with JSON fields: make, year, mileage, horsepower, fuel_type, body_type.</section>
</main>
</body>
</html>
"""


def options(values):
    return "".join(f'<option value="{value}">{value}</option>' for value in sorted(values))


@app.route("/", methods=["GET", "POST"])
def index():
    df = pd.read_csv(DATA_PATH)
    prediction = ""
    if request.method == "POST":
        payload = {feature: request.form[feature] for feature in FEATURES}
        payload["year"] = int(payload["year"])
        payload["mileage"] = int(payload["mileage"])
        payload["horsepower"] = int(payload["horsepower"])
        price = predict_price(payload)
        prediction = f"<section><div class='price'>${price:,.0f}</div><p>Estimated fair-market price</p></section>"
    return render_template_string(
        PAGE,
        make_options=options(df["make"].unique()),
        fuel_options=options(df["fuel_type"].unique()),
        body_options=options(df["body_type"].unique()),
        prediction=prediction,
    )


def predict_price(payload):
    bundle = load_bundle()
    query = pd.DataFrame([payload], columns=bundle["features"])
    return float(bundle["model"].predict(query)[0])


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    return jsonify({"estimated_price": round(predict_price(payload), 2)})


if __name__ == "__main__":
    app.run(debug=True)
