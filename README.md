# Customer Churn Prediction Project

This project aims to predict customer churn using machine learning techniques. It includes data preprocessing, feature engineering, model training, and deployment of the trained model through an API.

## Features

- End-to-end machine learning pipeline for customer churn prediction.
- Ensemble model using Random Forest and XGBoost.
- Automated model tracking and parameter logging via MLflow.
- Real-time and batch prediction APIs using FastAPI.
- Interactive dashboard for monitoring predictions.
- Dockerized deployment with CI/CD workflows on AWS EC2.
- Modular, production-ready code structure following best practices.

## Project Structure

```
customer-churn-prediction
├── data
│   ├── raw                # Directory for raw data files
│   ├── processed          # Directory for processed data files
│   └── sample_data.csv    # Sample dataset for customer churn prediction
├── src
│   ├── data
│   │   └── data_preprocessing.py  # Functions for data cleaning and transformation
│   ├── models
│   │   ├── ensemble_model.py       # Implementation of ensemble learning models
│   │   |── model_trainer.py        # Functions for training and evaluating models
|   |   └── predict_churn.py        # Functions for individual and batch churn predictions
│   ├── api
│   │   ├── main.py                  # Entry point for the API
│   │   └── schemas.py               # Data schemas for API requests and responses
|   ├── utils
|   │   └── logger.py
│   └── dashboard
│       └── dashboard.py             # Code for visualizations and dashboards
├── tests
│   ├── test_preprocessing.py         # Unit tests for data preprocessing functions
│   └── test_models.py                # Unit tests for model training and evaluation
├── config
│   ├── config.yaml                   # Configuration settings for the project
│   └── model_config.yaml             # Model-specific configuration settings
├── scripts
│   ├── train_model.py                # Script to train the machine learning model
│   ├── deploy.sh                     # Shell script for deploying the trained model
├── .github
│   └── workflows
│       ├── ci.yml                   # Continuous integration workflow configuration
│       └── cd.yml                   # Continuous deployment workflow configuration
├── Dockerfile                        # Instructions for building a Docker image
├── docker-compose.yml                # Defines services for running the project in Docker
├── requirements.txt                  # Lists Python dependencies required for the project
├── setup.py                          # Packaging script for the project
└── README.md                         # Documentation for the project
```

## Technologies Used
Programming Language: Python
Libraries: scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn
Model Serving: FastAPI
Dashboard: Streamlit
Tracking: MLflow
Testing: pytest
CI/CD: GitHub Actions
Containerization: Docker, docker-compose
Deployment: AWS EC2

## Model Performance
Accuracy: ~98%
Precision: ~97%
Recall: ~97%
F1-Score: ~97%

Ensemble of Random Forest and XGBoost.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Preprocess the data:
   - Use the `data_preprocessing.py` script to clean and transform the raw data.
   
2. Train the Model:
   - Run the `train_model.py` script to train the machine learning model.

3. Access the API:
   - The API can be accessed through the `main.py` script, which sets up the web server and routing.