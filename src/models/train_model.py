# src/models/train_model.py
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import mlflow
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/processed/dataset.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["train"]["test_size"], random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

mlflow.log_metric("rmse", rmse)

joblib.dump(model, "models/latest_model.pkl")
print("Modèle sauvegardé dans models/latest_model.pkl, RMSE:", rmse)

