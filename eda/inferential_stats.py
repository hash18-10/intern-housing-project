# eda/inferential_stats.py

import pandas as pd
import os
import logging
import scipy.stats as stats

# Ensure logs folder exists
os.makedirs("eda", exist_ok=True)

# Setup logger
logging.basicConfig(
    handlers=[logging.FileHandler("eda/inferential.log", mode='w')],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class InferentialStats:
    """
    Generate inferential insights from housing data.
    """

    def __init__(self, path: str):
        self.data_path = path
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            self.df.dropna(inplace=True)
            logging.info("Inferential data loaded and cleaned.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def correlation_summary(self):
        try:
            corr = self.df.corr(numeric_only=True)
            logging.info("Correlation Matrix:\n%s", corr.to_string())
            return corr
        except Exception as e:
            logging.error(f"Correlation analysis failed: {e}")

    def t_test_latitude(self):
        """
        Run a t-test comparing house prices north vs south (latitude > 35 vs <= 35)
        """
        try:
            north = self.df[self.df['latitude'] > 35]['median_house_value']
            south = self.df[self.df['latitude'] <= 35]['median_house_value']
            t_stat, p_val = stats.ttest_ind(north, south)
            logging.info(f"T-Test: t-stat = {t_stat:.2f}, p-value = {p_val:.4f}")
            return t_stat, p_val
        except Exception as e:
            logging.error(f"T-test failed: {e}")

    def run(self):
        self.load_data()
        self.correlation_summary()
        self.t_test_latitude()
        logging.info("Inferential analysis completed.")


if __name__ == "__main__":
    infer = InferentialStats("data/Intern Housing Data India.csv")
    infer.run()
