import pandas as pd
import numpy as np
import os
import requests
from io import StringIO

def download_churn_data():
    """
    Download and save sample customer churn data
    This is a well-known dataset from Kaggle
    """
    # Alternative data source - Kaggle dataset via raw.githubusercontent.com
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    try:
        # Download the data with SSL verification disabled for this attempt
        print("Downloading customer churn data...")
        
        # Method 1: Try with requests (more flexible)
        response = requests.get(url, verify=False)
        response.raise_for_status()  # Check if request was successful
        
        # Read the CSV content
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        
    except Exception as e:
        print(f"Error with URL download: {e}")
        print("Creating sample data instead...")
        # Create sample data as fallback
        df = create_sample_data()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)  # FIXED: Changed from '../data' to 'data'
    
    # Save the raw data
    raw_data_path = 'data/raw_churn_data.csv'  # FIXED: Changed from '../data' to 'data'
    df.to_csv(raw_data_path, index=False)
    
    print(f"Data saved to {raw_data_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def create_sample_data():
    """Create sample churn data if download fails"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customerID': [f'CUST{i:05d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(1, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'MonthlyCharges': np.round(np.random.uniform(20, 120, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(50, 8000, n_samples), 2),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75])
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    download_churn_data()