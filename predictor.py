import joblib
import pandas as pd
import os
from logger import logger

class CKDPredictor:
    def __init__(self, model_dir=None):
        # Resolve absolute path to the models directory relative to this script
        if model_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(base_dir, "models")
        else:
            self.model_dir = model_dir
            
        self.load_artifacts()

    def load_artifacts(self):
        try:
            model_path = os.path.join(self.model_dir, "best_ckd_model.pkl")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            cols_path = os.path.join(self.model_dir, "feature_columns.pkl")

            logger.info(f"Loading model from: {model_path}")

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = joblib.load(cols_path)
            logger.info("Artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise e

    def predict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        
        # Reorder and align columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]

        # Scale features
        scaled_data = self.scaler.transform(df)
        
        # Predict class and probabilities
        prediction = self.model.predict(scaled_data)[0]
        probabilities = self.model.predict_proba(scaled_data)[0]
        
        ckd_prob = probabilities[1] if len(probabilities) > 1 else float(prediction)
        
        if ckd_prob >= 0.70:
            risk_level = "High Risk"
        elif ckd_prob >= 0.35:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        return {
            "prediction": int(prediction),
            "probability": round(ckd_prob * 100, 2),
            "risk_level": risk_level
        }