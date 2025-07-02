import mlflow.sklearn
import pickle
import numpy as np

from flask import Flask, render_template_string, request
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

mlflow.set_tracking_uri('http://localhost:5000') 
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Prédiction Potabilité de l'Eau</title>
</head>
<body>
    <h1>Prédire la potabilité de l'eau</h1>
    <form method="post">
        {% for feature in features %}
            <label>{{ feature }}:</label>
            <input type="number" step="any" name="{{ feature }}" required><br><br>
        {% endfor %}
        <button type="submit">Prédire</button>
    </form>

    {% if prediction is not none %}
        <h2>Résultat de la prédiction : {{ 'Potable' if prediction == 1 else 'Non potable' }}</h2>
    {% endif %}
</body>
</html>
"""

FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

@app.route('/', methods=['GET', 'POST'])
def predict():
    model = mlflow.sklearn.load_model("models:/Waterflow XGBoost/latest")
    scaler = mlflow.sklearn.load_model("models:/Waterflow Scaler/latest")
    if not isinstance(model, XGBClassifier) or not isinstance(scaler, StandardScaler):
        return None
    prediction = None
    if request.method == 'POST':
        try:
            input_values = DataFrame({feature: [float(request.form[feature])] for feature in FEATURES})
            input_array = scaler.transform(input_values)
            prediction = model.predict(input_array)[0]
        except Exception as e:
            prediction = f"Erreur lors de la prédiction : {e}"

    return render_template_string(HTML_TEMPLATE, features=FEATURES, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
