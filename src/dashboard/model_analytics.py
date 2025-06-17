import pandas as pd
import json

from src.models.ensemble_model import EnsembleModel
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

class ModelAnalytics:
    def __init__(self, metrics_path='models/churn_model_metrics.json'):
        self.metrics_path = metrics_path
        self.model = EnsembleModel()
        self.model.load_model('models/churn_model.pkl')

    def get_model_metrics(self):
        with open(self.metrics_path, 'r') as f:
            metrics = json.load(f)
        rf_metrics = metrics['random_forest']
        xgb_metrics = metrics['xgboost']
        ensemble_metrics = metrics['ensemble']
        correlation_data = metrics['correlation_data']
        return rf_metrics, xgb_metrics, ensemble_metrics, correlation_data
    
    def get_feature_importances(self):
        model = EnsembleModel()
        model.load_model('models/churn_model.pkl')
        
        rf_importances = model.rf_model.feature_importances_
        xgb_importances = model.xgb_model.feature_importances_
        
        feature_names = model.rf_model.feature_names_in_
        logger.info(f"Feature names: {feature_names}")
        ensemble_importances = (model.ensemble_weights[0] * rf_importances 
                             + model.ensemble_weights[1] * xgb_importances)
        logger.info(f"Ensemble Importances: {ensemble_importances}")
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': ensemble_importances
        }).sort_values(by='importance', ascending=False)

        return feature_importance_df