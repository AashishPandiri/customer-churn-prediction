import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_trainer import ModelTrainer
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train the churn prediction model.")
    parser.add_argument('--data-path', required=True, help='Path to the training data')
    parser.add_argument('--model-path', default='models/churn_model.pkl', help='Path to save the trained model')
    parser.add_argument('--config-path', default='config/model_config.yaml', help='Path to the configuration file')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(config_path=args.config_path)
    
    logger.info("Starting model training...")
    model, metrics = trainer.train_pipeline(data_path=args.data_path, model_save_path=args.model_path)
    
    logger.info("Model training completed successfully.")
    logger.info(f"Model saved to {args.model_path}")
    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Model AUC: {metrics['auc']:.4f}")
    
if __name__ == "__main__":
    main()