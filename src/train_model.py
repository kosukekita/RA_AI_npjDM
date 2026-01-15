import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import os
import sys

# Add current directory to path to allow imports from config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import DATA_PATH, COL_OUTCOME, COL_HP_CODE, COL_PT_ID
except ImportError:
    # Fallback if running directly from src
    from .config import DATA_PATH, COL_OUTCOME, COL_HP_CODE, COL_PT_ID

def load_and_preprocess_data(filepath):
    """
    Loads data and performs basic preprocessing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}. Please update config.py.")

    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    
    # Filter valid outcomes
    df = df[df[COL_OUTCOME].isin([0, 1])].copy()
    
    return df

def train_catboost_model(df):
    """
    Trains the CatBoost model.
    """
    # Define features (all columns between 'sex' and 'drug_categ3_TNFi peg' excluding metadata)
    # Note: Adjust column selection logic based on your specific dataset structure
    try:
        start_col = 'sex'
        end_col = 'drug_categ3_TNFi peg'
        
        # Check if columns exist
        if start_col not in df.columns or end_col not in df.columns:
            # Fallback: use all columns except metadata
            cols = [c for c in df.columns if c not in [COL_OUTCOME, '6m_CDAI', COL_HP_CODE, COL_PT_ID]]
        else:
            cols = df.loc[:, start_col:end_col].columns.tolist()
            cols = [c for c in cols if c not in [COL_OUTCOME, '6m_CDAI', COL_HP_CODE, COL_PT_ID]]
    except Exception as e:
        print(f"Warning: Column selection failed ({e}). Using all non-metadata columns.")
        cols = [c for c in df.columns if c not in [COL_OUTCOME, '6m_CDAI', COL_HP_CODE, COL_PT_ID]]

    # Split train/test (Example: Using hospital code 'ky' as external validation)
    # Adjust this logic if you want a random split or different validation scheme
    train_df = df[df[COL_HP_CODE] != 'ky'].copy()
    test_df = df[df[COL_HP_CODE] == 'ky'].copy()
    
    X_train = train_df[cols]
    y_train = train_df[COL_OUTCOME]
    X_test = test_df[cols]
    y_test = test_df[COL_OUTCOME]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Model
    # Using default parameters as described in the paper (with 500 iterations)
    model = CatBoostClassifier(
        iterations=500, 
        verbose=0, 
        random_seed=42
    )
    
    print("Training CatBoost model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_proba)
    print(f"Model AUC on Validation Set: {auc:.4f}")
    
    return model, X_test, y_test, test_df, cols

if __name__ == "__main__":
    # Example usage
    try:
        df = load_and_preprocess_data(DATA_PATH)
        model, X_test, y_test, test_df, feature_names = train_catboost_model(df)
        print("Model training complete.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check config.py and ensure DATA_PATH is correct.")
