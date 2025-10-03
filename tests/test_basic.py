import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_data_files_exist():
    """Test that required data files exist"""
    assert os.path.exists('data/raw_churn_data.csv')
    assert os.path.exists('data/processed/X_train.csv')
    assert os.path.exists('data/processed/X_test.csv')

def test_model_file_exists():
    """Test that model file exists"""
    assert os.path.exists('models/random_forest_model.joblib')

def test_requirements_exist():
    """Test that requirements file exists"""
    assert os.path.exists('requirements.txt')