import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
APP_TITLE = "DrivenInsights Used Car Price Predictor"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown('''
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1180px;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.small-note {color: #5f6368; font-size: 0.92rem;}
</style>
''', unsafe_allow_html=True)


df = pd.read_csv("data/used_cars_sample.csv")
features = ["make", "year", "mileage", "horsepower", "fuel_type", "body_type"]
model = Pipeline([("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["make", "fuel_type", "body_type"])], remainder="passthrough")), ("rf", RandomForestRegressor(n_estimators=120, random_state=42))])
model.fit(df[features], df["price"])

st.title(APP_TITLE)
st.caption("Interactive fair-market estimate using a portfolio sample dataset modeled after the original PySpark ML workflow.")
col1, col2, col3 = st.columns(3)
make = col1.selectbox("Make", sorted(df.make.unique()))
year = col2.slider("Year", 2014, 2024, 2021)
mileage = col3.slider("Mileage", 0, 180000, 45000, step=1000)
col4, col5, col6 = st.columns(3)
hp = col4.slider("Horsepower", 100, 450, 210)
fuel = col5.selectbox("Fuel", sorted(df.fuel_type.unique()))
body = col6.selectbox("Body type", sorted(df.body_type.unique()))
query = pd.DataFrame([{"make": make, "year": year, "mileage": mileage, "horsepower": hp, "fuel_type": fuel, "body_type": body}])
pred = model.predict(query)[0]
st.metric("Estimated market price", f"${pred:,.0f}")

left, right = st.columns(2)
left.plotly_chart(px.scatter(df, x="mileage", y="price", color="body_type", hover_data=["make", "year"]), use_container_width=True)
right.plotly_chart(px.box(df, x="make", y="price", color="fuel_type"), use_container_width=True)
st.dataframe(df.sort_values("price", ascending=False), use_container_width=True, hide_index=True)
