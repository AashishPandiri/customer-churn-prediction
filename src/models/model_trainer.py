import yaml
import mlflow
import mlflow.sklearn
from src.data.data_preprocessing import DataPreprocessor
from src.models.ensemble_model import EnsembleModel

from src.utils.logger import setup_logger
import json
logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.preprocessor = DataPreprocessor()
        self.model = EnsembleModel(self.config.get('model_params', {}))
        
    def train_pipeline(self, data_path, model_save_path='models/churn_model.pkl'):
            
        df = self.preprocessor.load_data(data_path)
                
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_data(df)
        
        logger.info("Starting hyperparameter tuning...")
        self.model.tune_hyperparameters(X_train, y_train)
        logger.info("Hyperparameter tuning complete. Proceeding to training.")
        self.model.train(X_train, y_train)
        
        rf_metrics = self.model.evaluate(X_test, y_test, model_type='rf')
        xgb_metrics = self.model.evaluate(X_test, y_test, model_type='xgb')
        ensemble_metrics = self.model.evaluate(X_test, y_test)
        with mlflow.start_run():
            mlflow.log_params({f"rf_{key}": value for key, value in self.model.config['rf_params'].items()})
            mlflow.log_params({f"xgb_{key}": value for key, value in self.model.config['xgb_params'].items()})
            mlflow.log_metrics({'accuracy': ensemble_metrics['accuracy'],
                                'roc_auc': ensemble_metrics['auc'],
                                'f1_score': ensemble_metrics['classification_report']['weighted avg']['f1-score']})

            mlflow.sklearn.log_model(self.model, "ensemble_model")
        
        correlation_data = self.model.get_correlation_data(X_train)
        self.model.save_model(model_save_path)
        
        metrics_save_path = model_save_path.replace('.pkl', '_metrics.json')
        all_metrics = {
            'random_forest': rf_metrics,
            'xgboost': xgb_metrics,
            'ensemble': ensemble_metrics,
            'correlation_data': correlation_data
        }
        def make_serializable(metrics):
            import pandas as pd
            if isinstance(metrics, dict):
                return {k: make_serializable(v) for k, v in metrics.items()}
            elif isinstance(metrics, pd.DataFrame):
                return metrics.to_dict()
            else:
                return metrics
        with open(metrics_save_path, 'w') as f:
            json.dump(make_serializable(all_metrics), f, indent=4)
        logger.info(f"Saved model metrics to {metrics_save_path}")
        
        self.preprocessor.save_preprocessing(self.preprocessor.scaler, self.preprocessor.label_encoders)
        
        feature_importance_df = self.model.get_feature_importance(X_train.columns)
        logger.info("Top 10 most important features:")
        logger.info(feature_importance_df.head(10).to_string(index=False))
        
        return self.model, ensemble_metrics