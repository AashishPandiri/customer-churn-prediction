import pandas as pd
import numpy as np
from src.data.data_preprocessing import DataPreprocessor
from src.models.ensemble_model import EnsembleModel

class ChurnPredictor:
    def __init__(self, model_path='models/churn_model.pkl', preprocessing_prefix='models/preprocessing'):
        self.preprocessor = self._load_preprocessing(preprocessing_prefix)
        self.model = EnsembleModel()
        self.model.load_model(model_path)

    def predict_batch(self, data_path):
        df = self.preprocessor.load_data(data_path)
        df = self.preprocessor.clean_data(df)
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'], axis=1)
        df = self.preprocessor.encode_features(df, fit=False)
        X_scaled = self.preprocessor.scale_features(df, fit=False)
        probabilities = self.model.predict_proba(X_scaled)
        predictions = self.model.predict(X_scaled)
        return probabilities, predictions

    def predict_individual(self, customer_data):
        df = pd.DataFrame([customer_data])
        df = self.preprocessor.clean_data(df)
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'], axis=1)
        df = self.preprocessor.encode_features(df, fit=False)
        X_scaled = self.preprocessor.scale_features(df, fit=False)
        probabilities = self.model.predict_proba(X_scaled)
        prediction = self.model.predict(X_scaled)
        return probabilities[0], prediction[0]

    @staticmethod
    def _load_preprocessing(path_prefix='models/preprocessing'):
        preprocessor = DataPreprocessor()
        scaler, label_encoders = preprocessor.load_preprocessing(path_prefix)
        preprocessor.scaler = scaler
        preprocessor.label_encoders = label_encoders
        return preprocessor