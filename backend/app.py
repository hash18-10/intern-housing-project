# backend/app.py

from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import logging
import os

# Setup logger
os.makedirs("backend", exist_ok=True)
logging.basicConfig(
    handlers=[logging.FileHandler("backend/backend.log", mode='w')],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HousingAPI:
    """
    A class to handle prediction logic via API.
    """

    def __init__(self, model_path: str):
        """
        Initialize with path to trained model.
        """
        try:
            self.model = joblib.load(model_path)
            logging.info("Model loaded successfully in backend.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def predict(self, data: list) -> float:
        """
        Predict price from input features.
        """
        try:
            input_array = np.array(data).reshape(1, -1)
            price = self.model.predict(input_array)[0]
            logging.info(f"Prediction: {price} for input {data}")
            return price
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise


# Create Flask app
app = Flask(__name__)
model_api = HousingAPI(model_path="model/housing_model.pkl")


@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/inferential")
def inferential_page():
    return render_template("inferential.html")

@app.route("/descriptive")
def descriptive_page():
    return render_template("descriptive.html")




@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect data from form
        total_bedrooms = float(request.form["total_bedrooms"])
        households = float(request.form["households"])
        latitude = float(request.form["latitude"])
        longitude = float(request.form["longitude"])
        population = float(request.form["population"])

        # Prepare input and get prediction
        features = [total_bedrooms, households, latitude, longitude, population]
        predicted_price = model_api.predict(features)

        return render_template("predict.html", prediction=f"₹{predicted_price:,.2f}")
    except Exception as e:
        logging.error(f"Error in /predict route: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
