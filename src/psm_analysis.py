import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config as cfg
except ImportError:
    from . import config as cfg

def run_psm_analysis(df, covariates=None):
    """
    Performs Propensity Score Matching (PSM) analysis.
    
    Args:
        df: DataFrame containing 'concordant' (boolean/int) and 'CDAI_outcome'.
        covariates: List of column names to use for propensity score estimation.
    """
    print("Running Propensity Score Matching Analysis...")
    
    # Pre-checks
    if 'concordant' not in df.columns or cfg.COL_OUTCOME not in df.columns:
        raise ValueError(f"Dataframe must contain 'concordant' and '{cfg.COL_OUTCOME}' columns.")
        
    df = df.copy()
    df['concordant_int'] = df['concordant'].astype(int)
    
    # Default covariates if none provided
    if covariates is None:
        # Example covariates based on the paper
        covariates = [
            'age', 'sex', 'BMI', 'CDAI', 'SDAI', 'CRP', 'ESR1hr', 'RF', 
            'Treatment Line', 'stage', 'class', 'eGFR', 
            'disease_duration', 'Hb', 'Pt_VAS'
        ]
    
    # Filter available covariates
    available_covariates = [c for c in covariates if c in df.columns]
    print(f"Covariates used: {available_covariates}")
    
    # Prepare data for PS estimation
    X = df[available_covariates].copy()
    treatment = df['concordant_int'].values
    outcome = df[cfg.COL_OUTCOME].values
    
    # Impute missing values (Median imputation as per methods)
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
            
    # Estimate Propensity Scores
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X_scaled, treatment)
    ps = ps_model.predict_proba(X_scaled)[:, 1]
    df['propensity_score'] = ps
    
    # Matching (1:1 Nearest Neighbor with Caliper)
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    
    matched_treated = []
    matched_control = []
    used_controls = set()
    
    # Sort treated by PS
    treated_sorted = sorted(treated_idx, key=lambda i: ps[i])
    caliper = 0.2 * ps.std()
    
    for t_idx in treated_sorted:
        t_ps = ps[t_idx]
        available_controls = [c for c in control_idx if c not in used_controls]
        
        if not available_controls:
            break
            
        distances = [abs(ps[c] - t_ps) for c in available_controls]
        min_dist_idx = np.argmin(distances)
        
        if distances[min_dist_idx] <= caliper:
            best_control = available_controls[min_dist_idx]
            matched_treated.append(t_idx)
            matched_control.append(best_control)
            used_controls.add(best_control)
            
    print(f"Matched pairs: {len(matched_treated)}")
    
    # Analysis on Matched Data
    matched_outcomes_treated = outcome[matched_treated]
    matched_outcomes_control = outcome[matched_control]
    
    rate_treated = matched_outcomes_treated.mean()
    rate_control = matched_outcomes_control.mean()
    
    # Chi-square test
    n_matched = len(matched_treated)
    resp_treated = matched_outcomes_treated.sum()
    resp_control = matched_outcomes_control.sum()
    
    contingency = [[resp_treated, n_matched - resp_treated],
                   [resp_control, n_matched - resp_control]]
    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
    
    results = {
        'n_matched_pairs': n_matched,
        'concordant_response_rate': rate_treated,
        'discordant_response_rate': rate_control,
        'ARD': rate_treated - rate_control,
        'p_value': p_val
    }
    
    print("\nPSM Results:")
    print(f"  Concordant Rate: {rate_treated:.1%}")
    print(f"  Discordant Rate: {rate_control:.1%}")
    print(f"  ARD: {results['ARD']:.1%}")
    print(f"  p-value: {p_val:.4f}")
    
    return results

if __name__ == "__main__":
    pass
