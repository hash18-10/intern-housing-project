"""
Basic test to ensure the housing model loads properly.
"""

import joblib

def test_model_loading():
    model = joblib.load("model/housing_model.pkl")
    assert model is not None
