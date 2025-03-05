from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

import joblib

# Load the model globally to avoid repeated loading
model = load_model("wine_model.keras")
scaler=joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Wine Quality Prediction API</h1>"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Convert input to numpy array
        input_data = np.array(list(data.values())).reshape(1, -1)
        input=scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input) # Extract single value

        return jsonify({"Prediction_Value": prediction.tolist()}) # Convert to Python float for JSON serialization

    except ValueError:
        return jsonify({"error": "Invalid input data format"}), 400
    except KeyError:
        return jsonify({"error": "Missing required input values"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
