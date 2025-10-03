import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that we can import all required modules"""
    try:
        import pandas as pd
        import sklearn
        import fastapi
        import joblib
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_requirements_exist():
    """Test that requirements file exists"""
    assert os.path.exists('requirements.txt')

def test_src_files_exist():
    """Test that source files exist"""
    assert os.path.exists('src/app.py')
    assert os.path.exists('src/get_data.py')
    assert os.path.exists('src/process_data.py')
    assert os.path.exists('src/train_model.py')