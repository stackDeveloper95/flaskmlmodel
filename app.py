from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS 

# Load the trained model
with open("aluminium_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

CORS(app)

@app.route("/")
def home():
    return "Aluminium Strength Predictor API"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

       
        datas = pd.DataFrame(data["features"])

        predictions = model.predict(datas)

       
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
