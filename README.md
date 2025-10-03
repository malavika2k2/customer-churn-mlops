## Customer Churn Prediction - MLOps Project

A complete machine learning project that predicts customer churn with full MLOps practices.


### Run with Docker



docker build -t churn-api .
docker run -p 8000:8000 churn-api

Then visit: http://localhost:8000/docs

Run Locally:

# 1. Setup
git clone https://github.com/malavika2k2/customer-churn-mlops
cd customer-churn-mlops
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Get data & models
python src/get_data.py
dvc pull

# 3. Start API
uvicorn src.app:app --reload --port 8000


What This Project Does
----------------------

> Business Problem: Predict which customers will leave (churn)
> Solution: Machine learning model served via API
> Impact: Helps companies retain customers

Technologies Used
------------------

> ML: Scikit-learn, Pandas, NumPy
> API: FastAPI (auto docs at /docs)
> Tracking: MLflow (experiments at :5000)
> Data Versioning: DVC
> Container: Docker
> CI/CD: GitHub Actions
> Testing: pytest

Project Structure
------------------

src/
├── app.py           # FastAPI server
├── get_data.py      # Download data
├── process_data.py  # Clean & prepare data
└── train_model.py   # Train ML model

data/                # Customer data (DVC)
models/              # Trained models (DVC)
tests/               # Test cases


API Usage
---------

# Get predictions by sending customer data:


curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 70.50,
    "TotalCharges": 850.00,
    "SeniorCitizen": 0
  }'

# Response:

json

{
  "churn_probability": 0.23,
  "churn_prediction": false,
  "prediction_class": "No Churn"
}

Model Performance
----------------
>Accuracy: 72%
>Algorithm: Random Forest
>Key Features: Monthly charges, total charges, tenure

CI/CD Pipeline
-------------

# Every code change is automatically:
>Tested with pytest
>Built as Docker image
>Verified to work

Why This Project Matters
----------------------

# Demonstrates real-world MLOps skills:
* End-to-end ML pipeline
* Production-ready API
* Model versioning and tracking
* Automated testing and deployment
* Containerization

