# DrivenInsights Used Car Price Prediction

DrivenInsights is a machine learning project for estimating fair market value of used vehicles. The original coursework used PySpark models over a cleaned used-car dataset; this portfolio version preserves the notebook reference while adding a production-style Python training pipeline and Flask inference app. The original Spark workflow is also compatible with Azure Databricks-style notebook execution because it relies on standard Spark ML components.

## Technical Scope

The model uses vehicle make, model year, mileage, horsepower, fuel type, and body type to predict price. The training script compares Ridge Regression, Random Forest, and Gradient Boosting using R2, MAE, and RMSE, then serializes the best model with `joblib`.

This is intentionally more than a dashboard: it demonstrates model training, model selection, artifact persistence, and a lightweight API surface for prediction, while still reflecting a Spark/Databricks-ready analytics workflow in the original coursework.

## Files

- `notebooks/used_cars.ipynb` - original notebook reference from the coursework.
- `train_model.py` - reproducible scikit-learn training pipeline.
- `app.py` - Flask web app and JSON prediction API.
- `data/used_cars_sample.csv` - safe sample data for public execution.
- `models/` - generated locally when training runs; ignored by Git.

## Run Training

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
```

## Run Flask App

```powershell
python app.py
```

Open `http://127.0.0.1:5000`.

## API Example

```powershell
curl -X POST http://127.0.0.1:5000/predict `
  -H "Content-Type: application/json" `
  -d "{\"make\":\"Toyota\",\"year\":2021,\"mileage\":45000,\"horsepower\":210,\"fuel_type\":\"Hybrid\",\"body_type\":\"SUV\"}"
```

## Portfolio Note

The original dataset links are documented in `docs/Source.txt`. A compact sample dataset is included so reviewers can run the project without downloading large files.
