# model/predict_model.py

import joblib
import numpy as np
import logging
import os

# Setup logger
os.makedirs("model", exist_ok=True)
logging.basicConfig(
    handlers=[logging.FileHandler("model/predict.log", mode='w')],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HousingPricePredictor:
    """
    A class to load a trained model and make predictions for housing prices.
    """

    def __init__(self, model_path: str):
        """
        Initialize with path to the trained model.
        """
        try:
            self.model = joblib.load(model_path)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def predict_price(self, features: list) -> float:
        """
        Predict the price based on input features.

        Parameters:
            features (list): A list of input features in order:
                             [total_bedrooms, households, latitude, longitude, population]

        Returns:
            float: The predicted house price.
        """
        try:
            input_array = np.array(features).reshape(1, -1)
            prediction = self.model.predict(input_array)[0]
            logging.info(f"Prediction successful for input: {features}, Output: {prediction}")
            return prediction
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    try:
        predictor = HousingPricePredictor("model/housing_model.pkl")
        
        # Example test input [bedrooms, households, lat, long, population]
        sample_input = [3, 1200, 37.76, -122.42, 1500]
        predicted_price = predictor.predict_price(sample_input)
        
        print(f"Predicted house price: ₹{predicted_price:,.2f}")
    except Exception as e:
        print(f"Prediction failed: {e}")
