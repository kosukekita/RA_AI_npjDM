import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

def run_shap_analysis(model, X, save_dir='results'):
    """
    Runs SHAP analysis on the trained model.
    Generates Summary Plot and Feature Importance.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("Calculating SHAP values...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Handle CatBoost output (returns list if not binary specific, or depending on version)
    if isinstance(shap_values, list):
        # For binary classification, index 1 usually corresponds to the positive class
        shap_values = shap_values[1]
        
    # 1. Summary Plot
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, show=False, max_display=30)
    plt.title('SHAP Summary Plot', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance
    print("Generating Feature Importance...")
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Mean_SHAP': mean_shap_values
    }).sort_values('Mean_SHAP', ascending=False)
    
    csv_path = os.path.join(save_dir, 'shap_feature_importance.csv')
    feature_importance.to_csv(csv_path, index=False)
    print(f"Feature importance saved to {csv_path}")
    
    return shap_values, feature_importance

def plot_waterfall(model, X, row_idx, save_dir='results'):
    """
    Generates a waterfall plot for a specific patient.
    """
    explainer = shap.TreeExplainer(model)
    # Note: For waterfall plot, we often need the Explanation object
    explanation = explainer(X)
    
    plt.figure()
    shap.waterfall_plot(explanation[row_idx], show=False, max_display=15)
    plt.title(f'SHAP Waterfall Plot (Index: {row_idx})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'shap_waterfall_{row_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    pass
