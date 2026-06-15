import pytest
import pandas as pd
import numpy as np
from src.ui.utils import cramers_v, compute_risk_features

def test_cramers_v():
    # Perfect correlation
    x = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])
    y = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])
    assert np.isclose(cramers_v(x, y), 1.0)
    
    # Zero correlation approximation
    # Create perfectly independent distribution
    x = pd.Series(['A', 'A', 'B', 'B'])
    y = pd.Series(['C', 'D', 'C', 'D'])
    assert np.isclose(cramers_v(x, y), 0.0)

def test_compute_risk_features():
    df = pd.DataFrame({
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['No', 'No', 'No internet service'],
        'DeviceProtection': ['No', 'No', 'No internet service'],
        'InternetService': ['Fiber optic', 'DSL', 'No'],
        'StreamingTV': ['Yes', 'Yes', 'No internet service'],
        'StreamingMovies': ['Yes', 'No', 'No internet service'],
        'PaymentMethod': ['Electronic check', 'Credit card (automatic)', 'Mailed check'],
        'Contract': ['Month-to-month', 'One year', 'Two year']
    })
    
    df_feat = compute_risk_features(df)
    
    # Check if security_score is computed correctly
    assert df_feat['security_score'].tolist() == [2, 0, -1]
    
    # Check if streaming_score is computed correctly
    assert df_feat['streaming_score'].tolist() == [2, 1, -1]
    
    # Check if zero_supportive_service is computed correctly
    assert df_feat['zero_supportive_service'].tolist() == [0, 1, 0]
    
    # Check if manual_payment is computed correctly
    assert df_feat['manual_payment'].tolist() == [1, 0, 1]
    
    # Check if composite_risk_profile is computed correctly
    assert df_feat['composite_risk_profile'].tolist() == [1, 0, 0]

def test_compute_risk_features_missing_columns():
    df = pd.DataFrame({
        'InternetService': ['DSL', 'No']
    })
    
    df_feat = compute_risk_features(df)
    
    # If protective cols missing, should default to 0
    assert df_feat['security_score'].tolist() == [0, 0]
    
    # If streaming cols missing, should default to 0
    assert df_feat['streaming_score'].tolist() == [0, 0]
