# AI-Driven Treatment Recommendation System for Rheumatoid Arthritis

This repository contains the source code for the study: **"AI-Driven Treatment Recommendations for Rheumatoid Arthritis"** (submitted to npj Digital Medicine).

This code reproduces the machine learning pipeline, including:
1.  **Model Training**: Training a CatBoost model to predict treatment response (CDAI remission).
2.  **Treatment Recommendation**: An algorithm that combines ML predictions with clinical safety constraints (Rule-Based Filtering).
3.  **Statistical Analysis**: Propensity Score Matching (PSM) to evaluate the clinical utility of the recommendations.
4.  **Explainability**: SHAP analysis to interpret model decisions.

## Repository Structure

```
.
├── src/
│   ├── config.py              # Configuration (Paths, Thresholds, Drug Dictionaries)
│   ├── train_model.py         # CatBoost training script
│   ├── recommendation.py      # Recommendation algorithm (Prediction + Filtering)
│   ├── psm_analysis.py        # Propensity Score Matching implementation
│   └── explainability.py      # SHAP analysis scripts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Data Path**:
    Open `src/config.py` and set the `DATA_PATH` to point to your clinical dataset.
    The dataset should be an Excel or CSV file containing the clinical variables described in the paper (e.g., eGFR, WBC, Age, Drug History, etc.).

## Usage

### 1. Train the Model
To train the CatBoost model on your data:
```python
from src.train_model import load_and_preprocess_data, train_catboost_model
from src.config import DATA_PATH

df = load_and_preprocess_data(DATA_PATH)
model, X_test, y_test, test_df, feature_names = train_catboost_model(df)
```

### 2. Generate Recommendations
To generate optimal treatment recommendations for patients using the "AI Prediction + Safety Filtering" logic:
```python
from src.recommendation import TreatmentRecommender

recommender = TreatmentRecommender(model, feature_names)
recommendations_df = recommender.generate_recommendations(test_df)

# Merge recommendations back to test data
test_df_with_rec = test_df.join(recommendations_df)
```

### 3. Evaluate (Concordant vs Discordant)
To perform Propensity Score Matching (PSM) analysis comparing patients who followed the recommendation (Concordant) vs those who did not (Discordant):
```python
from src.psm_analysis import run_psm_analysis

# Define concordance
test_df_with_rec['concordant'] = test_df_with_rec['actual_category'] == test_df_with_rec['recommended_category']

# Run PSM
psm_results = run_psm_analysis(test_df_with_rec)
```

### 4. Interpretability (SHAP)
To generate SHAP summary plots and waterfall plots:
```python
from src.explainability import run_shap_analysis

run_shap_analysis(model, X_test)
```

## Methodology Notes

*   **Model**: CatBoostClassifier (500 iterations).
*   **Filtering Logic**: The recommendation system strictly adheres to clinical guidelines (e.g., MTX contraindications for eGFR < 30, JAKi exclusion for patients with multiple risk factors). See `src/recommendation.py` for details.
*   **PSM**: 1:1 Nearest Neighbor matching with a caliper of 0.2 * SD of propensity score.

## License

[License Information Here]
