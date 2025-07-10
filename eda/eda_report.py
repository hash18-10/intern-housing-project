# eda/eda_report.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

# Set up logger

os.makedirs('eda', exist_ok=True)
# Set up logging to write to eda.log inside the eda folder
logging.basicConfig(
    handlers=[logging.FileHandler("eda/eda.log", mode='w')],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HousingEDA:
    """
    A class to perform Exploratory Data Analysis on the Intern Housing dataset.
    """

    def __init__(self, filepath: str):
        """
        Initialize with the path to the dataset.
        """
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """
        Load the dataset into a pandas DataFrame.
        """
        try:
            self.df = pd.read_csv(self.filepath)
            logging.info("Dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    def basic_info(self):
        """
        Log dataset shape, head, and missing value count.
        """
        logging.info(f"Data Shape: {self.df.shape}")
        logging.info(f"Missing Values:\n{self.df.isnull().sum()}")
        logging.info(f"Data Description:\n{self.df.describe()}")

    def plot_distribution(self):
        """
        Plot and save distribution of house prices.
        """
        try:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df['median_house_value'], bins=30, kde=True)
            plt.title("Distribution of House Prices")
            plt.xlabel("Price")
            plt.ylabel("Frequency")
            plt.tight_layout()
            dist_path = 'eda/price_distribution.png'
            plt.savefig(dist_path)
            plt.close()
            logging.info(f"Price distribution plot saved to {dist_path}")
        except Exception as e:
            logging.error(f"Error plotting distribution: {e}")

    def plot_correlation_heatmap(self):
        """
        Plot and save a heatmap of feature correlations.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            heatmap_path = 'eda/correlation_heatmap.png'
            plt.savefig(heatmap_path)
            plt.close()
            logging.info(f"Correlation heatmap saved to {heatmap_path}")
        except Exception as e:
            logging.error(f"Error plotting correlation heatmap: {e}")

    def run_all(self):
        """
        Run the full EDA pipeline.
        """
        self.load_data()
        self.basic_info()
        self.plot_distribution()
        self.plot_correlation_heatmap()


if __name__ == "__main__":
    # Ensure eda output directory exists
    os.makedirs("eda", exist_ok=True)

    eda = HousingEDA(filepath="data/Intern Housing Data India.csv")
    eda.run_all()
    logging.info("EDA pipeline completed.")
