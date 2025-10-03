import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_and_explore_data():
    """Load the raw data and explore its structure"""
    data_path = 'data/raw_churn_data.csv'  # CORRECT PATH
    df = pd.read_csv(data_path)
    
    print("=== DATA EXPLORATION ===")
    print(f"Dataset shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nChurn distribution:")
    print(df['Churn'].value_counts())
    
    return df

def preprocess_data(df):
    """Clean and preprocess the data for modeling"""
    print("\n=== PREPROCESSING DATA ===")
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Convert Churn to binary (Yes=1, No=0)
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    # Select features for initial model - using numeric columns for simplicity
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    
    # Prepare features and target
    X = data[numeric_features]
    y = data['Churn']
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    print(f"Features used: {numeric_features}")
    
    return X, y

def split_and_save_data(X, y):
    """Split data into train/test and save processed versions"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create processed data directory
    os.makedirs('data/processed', exist_ok=True)  # CORRECT PATH
    
    # Save processed data
    X_train.to_csv('data/processed/X_train.csv', index=False)  # CORRECT PATH
    X_test.to_csv('data/processed/X_test.csv', index=False)    # CORRECT PATH
    y_train.to_csv('data/processed/y_train.csv', index=False)  # CORRECT PATH
    y_test.to_csv('data/processed/y_test.csv', index=False)    # CORRECT PATH
    
    print("\n=== DATA SPLITTING ===")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Training churn distribution: {y_train.value_counts()}")
    print(f"Test churn distribution: {y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load and explore
    df = load_and_explore_data()
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split and save
    split_and_save_data(X, y)
    
    print("\n Data processing complete!")
    print("Processed files saved to data/processed/")