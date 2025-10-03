import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
import joblib
import os

def load_processed_data():
    """Load the processed training and test data"""
    print("Loading processed data...")
    
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train a Random Forest classifier and track with MLflow"""
    
    # Start MLflow experiment
    mlflow.set_experiment("customer_churn_prediction")
    
    with mlflow.start_run():
        print("Training Random Forest model...")
        
        # Create and train the model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f}")
        
        # Log parameters to MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("cv_mean_score", cv_scores.mean())
        mlflow.log_metric("cv_std_score", cv_scores.std())
        
        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Log feature importance
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        # Print feature importance
        print("\nFeature Importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, accuracy

def save_model_locally(model, accuracy):
    """Save the trained model locally"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = 'models/random_forest_model.joblib'
    joblib.dump(model, model_path)
    
    print(f"\nModel saved locally to: {model_path}")
    print(f"Model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    print("=== MODEL TRAINING WITH MLFLOW ===")
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model with MLflow tracking
    model, accuracy = train_random_forest(X_train, X_test, y_train, y_test)
    
    # Save model locally
    save_model_locally(model, accuracy)
    
    print("\n Model training complete")
    print("Check MLflow UI to see your experiment: mlflow ui")