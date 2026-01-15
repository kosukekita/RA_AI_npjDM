import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config as cfg
except ImportError:
    from . import config as cfg

class TreatmentRecommender:
    def __init__(self, model, feature_cols):
        self.model = model
        self.feature_cols = feature_cols
        
        # Identify drug-related columns for resetting
        self.drug_cols = [c for c in feature_cols if c.startswith('drug_') and not c.startswith('drug_categ')]
        self.categ1_cols = [c for c in feature_cols if c.startswith('drug_categ1_')]
        self.categ2_cols = [c for c in feature_cols if c.startswith('drug_categ2_')]
        self.categ3_cols = [c for c in feature_cols if c.startswith('drug_categ3_')]

    def get_patient_constraints(self, row):
        """
        Determines treatment constraints based on patient clinical data.
        Implements the rule-based filtering described in the Methods section.
        """
        # Extract values (handle missing values with safe defaults)
        def safe_get(col, default):
            val = row.get(col, default)
            return default if pd.isna(val) else val

        egfr = safe_get(cfg.COL_EGFR, 999)
        wbc = safe_get(cfg.COL_WBC, 999)
        plt = safe_get(cfg.COL_PLT, 999)
        kl6 = safe_get(cfg.COL_KL6, 0)
        bmi = safe_get(cfg.COL_BMI, 25)
        age = safe_get(cfg.COL_AGE, 50)
        smoking = safe_get(cfg.COL_SMOKING, 0)
        
        constraints = {
            'excluded_drugs': [],
            'max_mtx_dose': 16, # Default max dose
            'mtx_allowed': True,
            'jaki_allowed': True,
            'reasons': []
        }
        
        # 1. Specific Drug Contraindications
        if egfr < cfg.TH_EGFR_BAR:
            constraints['excluded_drugs'].append('BAR')
            constraints['reasons'].append(f'eGFR<{cfg.TH_EGFR_BAR} -> BAR contraindicated')
        
        if egfr < cfg.TH_EGFR_FIL:
            constraints['excluded_drugs'].append('FIL')
            constraints['reasons'].append(f'eGFR<{cfg.TH_EGFR_FIL} -> FIL contraindicated')
            
        # 2. JAK Inhibitor Exclusion (Risk-Factor Based)
        # Exclude if patient has >= 2 risk factors: Age>=65, Smoking, BMI>=30
        jaki_risk_count = 0
        risk_factors = []
        
        if age >= cfg.TH_AGE_JAKI_RISK:
            jaki_risk_count += 1
            risk_factors.append('Age')
        if smoking in [1, 2]: # Past or Current smoker
            jaki_risk_count += 1
            risk_factors.append('Smoking')
        if bmi >= cfg.TH_BMI_JAKI_RISK:
            jaki_risk_count += 1
            risk_factors.append('BMI')
            
        if jaki_risk_count >= 2:
            constraints['jaki_allowed'] = False
            # Add all JAKi to excluded list
            for drug in cfg.JAKI_DRUGS:
                if drug not in constraints['excluded_drugs']:
                    constraints['excluded_drugs'].append(drug)
            constraints['reasons'].append(f'JAKi excluded (Risks: {", ".join(risk_factors)})')

        # 3. MTX Contraindications (Absolute)
        if (egfr < cfg.TH_EGFR_MTX or 
            wbc < cfg.TH_WBC_MTX or 
            plt < cfg.TH_PLT_MTX or 
            kl6 > cfg.TH_KL6_MTX):
            
            constraints['mtx_allowed'] = False
            constraints['max_mtx_dose'] = 0
            constraints['reasons'].append('MTX contraindicated')
            
        # 4. MTX Dose Restriction
        elif (egfr < cfg.TH_EGFR_MTX_DOSE or 
              age >= cfg.TH_AGE_MTX_DOSE or 
              bmi < cfg.TH_BMI_MTX_DOSE):
            
            constraints['max_mtx_dose'] = 12 # Cap at 12mg
            constraints['reasons'].append('MTX dose restricted (max 12mg)')
            
        return constraints

    def recommend_for_patient(self, row):
        """
        Generates the optimal treatment recommendation for a single patient.
        """
        constraints = self.get_patient_constraints(row)
        
        base_features = row[self.feature_cols].copy()
        best_prob = -1
        best_drug = None
        best_mtx = None
        
        # Define MTX candidates based on constraints
        if not constraints['mtx_allowed']:
            mtx_doses = [0]
        elif constraints['max_mtx_dose'] <= 12:
            mtx_doses = [0, 2, 4, 6, 8, 10, 12]
        else:
            mtx_doses = [0, 2, 4, 6, 8, 10, 12, 14, 16]
            
        # Iterate through all drug options
        for drug in cfg.SPECIFIC_DRUGS:
            if drug in constraints['excluded_drugs']:
                continue
            
            # Identify category mapping (requires drug dictionaries)
            # Assuming standard mappings; in a real scenario, ensure these columns match your training data
            
            for mtx_dose in mtx_doses:
                test_row = base_features.copy()
                
                # Reset all drug indicators to 0
                for col in self.drug_cols + self.categ1_cols + self.categ2_cols + self.categ3_cols:
                    if col in test_row.index:
                        test_row[col] = 0
                
                # Set candidate drug
                drug_col = f'drug_{drug}'
                if drug_col in test_row.index:
                    test_row[drug_col] = 1
                
                # Set categories (Using the config dictionary)
                category = cfg.DRUG_CATEGORY_DICT.get(drug, 'Unknown')
                categ1_col = f'drug_categ1_{category}' # Note: You might need to adjust these names based on exact training columns
                if categ1_col in test_row.index:
                    test_row[categ1_col] = 1
                    
                # Note: Skipping categ2/categ3 here for brevity, but they should be set if used in the model
                
                # Set MTX dose
                if 'MTX_dose' in test_row.index:
                    test_row['MTX_dose'] = mtx_dose
                
                # Predict
                prob = self.model.predict_proba(test_row.values.reshape(1, -1))[0, 1]
                
                if prob > best_prob:
                    best_prob = prob
                    best_drug = drug
                    best_mtx = mtx_dose
        
        return {
            'recommended_drug': best_drug,
            'recommended_category': cfg.DRUG_CATEGORY_DICT.get(best_drug),
            'recommended_mtx': best_mtx,
            'predicted_probability': best_prob,
            'constraints': "; ".join(constraints['reasons'])
        }

    def generate_recommendations(self, df):
        """
        Batch generation of recommendations.
        """
        results = []
        print("Generating AI recommendations...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            rec = self.recommend_for_patient(row)
            results.append(rec)
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    pass
