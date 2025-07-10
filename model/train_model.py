# model/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging

# Set up logging
os.makedirs("model", exist_ok=True)
logging.basicConfig(
    handlers=[logging.FileHandler("model/train.log", mode='w')],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class HousingModelTrainer:
    """
    A class to handle data cleaning, model training, evaluation, and saving.
    """

    def __init__(self, data_path: str):
        """
        Initialize with the path to the dataset.
        """
        self.data_path = data_path
        self.df = None
        self.model = None

    def load_and_clean_data(self):
        """
        Load the CSV file and clean it (handle missing values).
        """
        try:
            self.df = pd.read_csv(self.data_path)
            logging.info("Dataset loaded successfully.")
            
            # Fill missing values with median
            self.df.fillna(self.df.median(numeric_only=True), inplace=True)
            logging.info("Missing values handled.")
        except Exception as e:
            logging.error(f"Error loading/cleaning data: {e}")
            raise

    def train_model(self):
        """
        Train a Linear Regression model using selected features.
        """
        try:
            X = self.df[['total_bedrooms', 'households', 'latitude', 'longitude', 'population']]
            y = self.df['median_house_value']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.model = LinearRegression()
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model trained. MSE: {mse:.2f}, R2 Score: {r2:.2f}")
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise

    def save_model(self, model_path='model/housing_model.pkl'):
        """
        Save the trained model to disk.
        """
        try:
            joblib.dump(self.model, model_path)
            logging.info(f"Model saved to {model_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def run_pipeline(self):
        """
        Full pipeline: load data, clean, train, and save model.
        """
        self.load_and_clean_data()
        self.train_model()
        self.save_model()


if __name__ == "__main__":
    trainer = HousingModelTrainer(data_path='data/Intern Housing Data India.csv')
    trainer.run_pipeline()
    logging.info("Training pipeline completed.")
