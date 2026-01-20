"""
Configuration file for RA Treatment Response Prediction Project.

This file contains:
1. File paths (placeholders)
2. Column definitions
3. Drug dictionaries
4. Clinical thresholds
"""

# --- File Paths ---
# Please set the path to your dataset (Excel or CSV)
DATA_PATH = "path/to/your/dataset.xlsx" 

# Directory to save results
RESULT_DIR = "results"

# --- Column Names ---
# Define the column names used in your dataset
COL_OUTCOME = 'CDAI_outcome'  # Binary outcome (1: Responder if 6-month CDAI ≤ 10, 0: Non-Responder)

# --- Treatment Response Threshold ---
# Responder is defined as achieving Low Disease Activity (LDA) at 6 months: CDAI ≤ 10
TH_CDAI_RESPONSE = 10
COL_HP_CODE = 'hp_code'       # Hospital code for splitting train/test
COL_PT_ID = 'PT_ID'           # Patient ID

# Clinical variable columns (for filtering constraints)
COL_EGFR = 'eGFR'
COL_WBC = 'WBC'
COL_PLT = 'Plt'
COL_KL6 = 'KL-6'
COL_BMI = 'BMI'
COL_AGE = 'age'
COL_SMOKING = 'smoking' # 0: None, 1: Past, 2: Current

# --- Drug Definitions ---
# List of biological/targeted synthetic DMARDs
SPECIFIC_DRUGS = [
    'ABT', 'ADA', 'ADA-BS', 'BAR', 'CZP', 'ETN', 'ETN-BS', 
    'FIL', 'GLM', 'IFX', 'IFX-BS', 'OZR', 'PEF', 'SAR', 
    'TCZ', 'TOF', 'UPA'
]

# JAK Inhibitors
JAKI_DRUGS = ['BAR', 'FIL', 'OZR', 'PEF', 'TOF', 'UPA']

# Drug Categories Mapping
DRUG_CATEGORY_DICT = {
    'ABT': 'CTLA4', 'ADA': 'TNFi', 'ADA-BS': 'TNFi', 'BAR': 'JAKi',
    'CZP': 'TNFi', 'ETN': 'TNFi', 'ETN-BS': 'TNFi', 'FIL': 'JAKi',
    'GLM': 'TNFi', 'IFX': 'TNFi', 'IFX-BS': 'TNFi', 'OZR': 'JAKi',
    'PEF': 'JAKi', 'SAR': 'IL-6i', 'TCZ': 'IL-6i', 'TOF': 'JAKi', 'UPA': 'JAKi'
}

# --- Clinical Constraints Thresholds ---
# MTX Contraindications
TH_EGFR_MTX = 30
TH_WBC_MTX = 3
TH_PLT_MTX = 5
TH_KL6_MTX = 1000

# MTX Dose Restrictions (Max 12mg)
TH_EGFR_MTX_DOSE = 60
TH_AGE_MTX_DOSE = 75
TH_BMI_MTX_DOSE = 18.5

# Drug Specific Contraindications
TH_EGFR_BAR = 30
TH_EGFR_FIL = 15

# JAKi Risk Factors (Age, Smoking, BMI) -> Exclusion if >= 2 factors
TH_AGE_JAKI_RISK = 65
TH_BMI_JAKI_RISK = 30
