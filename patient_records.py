import pandas as pd
import os
from datetime import datetime
from logger import logger

CSV_FILE = "patient_records.csv"

def initialize_records_csv():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["Timestamp", "Patient_ID", "Age", "BP", "Creatinine", "Risk_Level", "CKD_Probability_%"])
        df.to_csv(CSV_FILE, index=False)
        logger.info("Initialized patient_records.csv")

def save_patient_record(patient_id, age, bp, creatinine, risk_level, prob):
    initialize_records_csv()
    new_record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Patient_ID": patient_id,
        "Age": age,
        "BP": bp,
        "Creatinine": creatinine,
        "Risk_Level": risk_level,
        "CKD_Probability_%": prob
    }
    df = pd.DataFrame([new_record])
    df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    logger.info(f"Saved record for Patient ID: {patient_id}")

def get_all_records():
    initialize_records_csv()
    return pd.read_csv(CSV_FILE)