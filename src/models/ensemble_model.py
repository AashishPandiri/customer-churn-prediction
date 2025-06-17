import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn

from src.utils.logger import setup_logger
logger = setup_logger(__name__)

class EnsembleModel:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.rf_model = None
        self.xgb_model = None
        self.ensemble_weights = [0.5, 0.5]
        
    def _default_config(self):
        return {
            'rf_params': {
                'n_estimators': 500,
                'max_leaf_nodes': 16,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'xgb_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        }
        
    def handle_imbalanced_data(self, X, y):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"Original class distribution: {np.bincount(y)}")
        logger.info(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train(self, X_train, y_train, handle_imabalance=True):
        logger.info("Starting model training process...")
        
        if handle_imabalance:
            X_train, y_train = self.handle_imbalanced_data(X_train, y_train)
            
        logger.info("Training Random Forest model...")
        self.rf_model = RandomForestClassifier(**self.config['rf_params'])
        self.rf_model.fit(X_train, y_train)
        
        logger.info("Training XGBoost model...")
        self.xgb_model = XGBClassifier(**self.config['xgb_params'])
        self.xgb_model.fit(X_train, y_train)
        
        logger.info("Model training completed.")
        
    def predict_proba(self, X):
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        
        ensemble_proba = (self.ensemble_weights[0] * rf_proba 
                          + self.ensemble_weights[1] * xgb_proba)
        
        return ensemble_proba

    def predict(self, X, threshold=0.5):
        ensemble_proba = self.predict_proba(X)
        return (ensemble_proba >= threshold).astype(int)
    
    def evaluate(self, X_test, y_test, model_type=None):
        if model_type == 'rf':
            y_pred = self.rf_model.predict(X_test)
            y_proba = self.rf_model.predict_proba(X_test)[:, 1]
            model_name = 'Random Forest'
        elif model_type == 'xgb':
            y_pred = self.xgb_model.predict(X_test)
            y_proba = self.xgb_model.predict_proba(X_test)[:, 1]
            model_name = 'XGBoost'
        else:
            y_pred = self.predict(X_test)
            y_proba = self.predict_proba(X_test)
            model_name = 'Ensemble'
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        logger.info(f"{model_name} Evaluation - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        return {
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
    def get_feature_importance(self, feature_names):
        rf_importance = self.rf_model.feature_importances_
        xgb_importance = self.xgb_model.feature_importances_
        
        combined_importance = (self.ensemble_weights[0] * rf_importance 
                              + self.ensemble_weights[1] * xgb_importance)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': combined_importance
        }).sort_values(by='importance', ascending=False)
        
        logger.info("Feature importance calculated.")
        return feature_importance
    
    def get_correlation_data(self, X):
        correlation_matrix = X.corr()
        logger.info("Correlation matrix calculated.")
        return correlation_matrix
    
    def save_model(self, model_path):
        model_data = {
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'ensemble_weights': self.ensemble_weights,
            'config': self.config
        }
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path):
        model_data = joblib.load(model_path)
        self.rf_model = model_data['rf_model']
        self.xgb_model = model_data['xgb_model']
        self.ensemble_weights = model_data['ensemble_weights']
        self.config = model_data['config']
        
        logger.info(f"Model loaded from {model_path}")
        
    def tune_hyperparameters(self, X, y):
        from sklearn.model_selection import GridSearchCV
        logger.info("Tuning Random Forest hyperparameters...")
        rf_grid = {
            'n_estimators': [100, 200, 500],
            'max_leaf_nodes': [10, 16, 20],
            'max_depth': [6, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        rf = RandomForestClassifier(random_state=42)
        rf_search = GridSearchCV(rf, rf_grid, cv=3, scoring='accuracy', n_jobs=-1)
        rf_search.fit(X, y)
        self.config['rf_params'] = rf_search.best_params_
        logger.info(f"Best RF params: {rf_search.best_params_}")

        logger.info("Tuning XGBoost hyperparameters...")
        xgb_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_search = GridSearchCV(xgb, xgb_grid, cv=3, scoring='accuracy', n_jobs=-1)
        xgb_search.fit(X, y)
        self.config['xgb_params'] = xgb_search.best_params_
        logger.info(f"Best XGB params: {xgb_search.best_params_}")