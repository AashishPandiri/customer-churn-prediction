import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import yaml


from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
    def clean_data(self, df):
        df = df.dropna(subset=['customerID'])
        
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        for feature in numerical_features:
            df[feature] = df[feature].fillna(df[feature].median())
            
        categorical_features = df.select_dtypes(include=[object]).columns.tolist()
        for feature in categorical_features:
            if feature != 'customerID':
                df[feature] = df[feature].fillna(df[feature].mode()[0])

        df = df.drop_duplicates(subset=['customerID'])

        logger.info(f"Data cleaned. Final Shape: {df.shape}")
        return df
    
    def encode_features(self, df, fit=True):
        categorical_features = df.select_dtypes(include=[object]).columns.tolist()
        
        for feature in categorical_features:
            if feature in df.columns:
                if fit:
                    label_encoder = LabelEncoder()
                    df[feature] = label_encoder.fit_transform(df[feature])
                    self.label_encoders[feature] = label_encoder
                else:
                    if feature in self.label_encoders:
                        df[feature] = self.label_encoders[feature].transform(df[feature].astype(str))
                        
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].astype(str).str.strip()
            logger.info(f"Unique values in Churn before mapping: {df['Churn'].unique()}")
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
        logger.info("Categorical features encoded.")
        return df
            
    def scale_features(self, X, fit=True):
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        logger.info("Features scaled.")
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def preprocess_data(self, df, target_column='Churn', test_size=0.2, random_state=42):
        df = self.clean_data(df)
        df = self.encode_features(df, fit=True)
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'], axis=1)
        df = df.dropna(subset=[target_column])
        logger.info(f"Rows remaining after dropping NaN in {target_column}: {len(df)}")
        if df.empty:
            raise ValueError("No data left after dropping rows with NaN in the target column.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_test_scaled = self.scale_features(X_test, fit=False)
        logger.info("Data preprocessing completed.")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessing(self, scaler, label_encoders, path_prefix='models/preprocessing'):
        import joblib
        joblib.dump(scaler, f'{path_prefix}_scaler.pkl')
        joblib.dump(label_encoders, f'{path_prefix}_encoders.pkl')
        logger.info(f"Saved scaler and encoders to {path_prefix}_scaler.pkl and {path_prefix}_encoders.pkl")

    def load_preprocessing(self, path_prefix='models/preprocessing'):
        import joblib
        scaler = joblib.load(f'{path_prefix}_scaler.pkl')
        label_encoders = joblib.load(f'{path_prefix}_encoders.pkl')
        logger.info(f"Loaded scaler and encoders from {path_prefix}_scaler.pkl and {path_prefix}_encoders.pkl")
        return scaler, label_encoders