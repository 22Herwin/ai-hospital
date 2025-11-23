import numpy as np
import pandas as pd
import argparse
import random
import math
from typing import Dict, Any, Optional

# Diagnoses and medicine mapping consistent with the app
DIAGNOSIS_MAPPING = {
    # Long Inpatient (7+ days)
    'D01': {
        'name': 'Heart Failure Exacerbation',
        'medicines': ['Furosemide 40mg IV', 'Carvedilol 6.25mg', 'Spironolactone 25mg']
    },
    'D02': {
        'name': 'COPD Exacerbation with Respiratory Failure',
        'medicines': ['Methylprednisolone 40mg IV', 'Levofloxacin 750mg', 'Albuterol Nebulizer']
    },
    'D03': {
        'name': 'Sepsis with Multi-organ Dysfunction',
        'medicines': ['Vancomycin 15mg/kg IV', 'Piperacillin-Tazobactam 4.5g IV', 'Norepinephrine infusion']
    },
    'D04': {
        'name': 'Stroke with Rehabilitation Needs',
        'medicines': ['Aspirin 325mg', 'Atorvastatin 80mg', 'Clopidogrel 75mg']
    },
    'D05': {
        'name': 'Pancreatitis with Complications',
        'medicines': ['Pantoprazole 40mg IV', 'Morphine PCA', 'Octreotide infusion']
    },
    # Short Inpatient (1-3 days)
    'D06': {
        'name': 'Pneumonia (Community Acquired)',
        'medicines': ['Ceftriaxone 1g IV', 'Azithromycin 500mg IV', 'Oxygen therapy']
    },
    'D07': {
        'name': 'UTI with Sepsis',
        'medicines': ['Ceftriaxone 2g IV', 'Gentamicin 5mg/kg IV', 'IV Fluids']
    },
    'D08': {
        'name': 'Hypertensive Crisis',
        'medicines': ['Labetalol infusion', 'Nicardipine drip', 'Hydralazine 10mg IV']
    },
    'D09': {
        'name': 'Diabetic Ketoacidosis (Resolved)',
        'medicines': ['Insulin infusion', 'Potassium replacement', 'IV Fluids']
    },
    'D10': {
        'name': 'Cellulitis with Systemic Symptoms',
        'medicines': ['Vancomycin 15mg/kg IV', 'Clindamycin 600mg IV', 'Wound care']
    },
    # Outpatient Care
    'D11': {
        'name': 'Upper Respiratory Infection',
        'medicines': ['Paracetamol 500mg', 'Dextromethorphan 15mg', 'Saline nasal spray']
    },
    'D12': {
        'name': 'Hypertension Management',
        'medicines': ['Amlodipine 5mg', 'Lisinopril 10mg', 'Hydrochlorothiazide 12.5mg']
    },
    'D13': {
        'name': 'Type 2 Diabetes Follow-up',
        'medicines': ['Metformin 1000mg', 'Sitagliptin 100mg', 'Glipizide 5mg']
    },
    'D14': {
        'name': 'Gastroenteritis',
        'medicines': ['Ondansetron 4mg', 'Loperamide 2mg', 'Oral Rehydration Salts']
    },
    'D15': {
        'name': 'Back Pain',
        'medicines': ['Ibuprofen 400mg', 'Cyclobenzaprine 5mg', 'Acetaminophen 650mg']
    },
    # Healthy
    'D16': {
        'name': 'Well Visit',
        'medicines': []
    }
}

def engineered_flags(row: Dict[str, Any]) -> Dict[str, int]:
    """
    Create engineered boolean/int flags to help ML separate disease patterns.
    These features will be used in both dataset generation and model training.
    """
    flags = {}
    flags['is_respiratory'] = int((row['symptom_cough'] == 1) or (row['symptom_breathless'] == 1))
    flags['is_infection'] = int((row['symptom_fever'] == 1) and (row['lab_crp'] > 20 or row['lab_wbc'] > 11))
    flags['is_cardiac'] = int((row['comorbidity_hypertension'] == 1) or (row['blood_pressure_sys'] > 160) or (row['heart_rate'] > 110))
    flags['is_metabolic'] = int(row['comorbidity_diabetes'] == 1)
    flags['is_neuro'] = int(row.get('neuro_flag', 0))
    flags['is_gi'] = int(row['diagnosis_code'] in ['D05', 'D14'])
    flags['age_over_65'] = int(row['age'] > 65)
    return flags

def gen_row(i: int, category_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a single patient row. Optionally force the 'category' via category_override.
    """
    # Choose category
    categories = ['long_inpatient', 'short_inpatient', 'outpatient', 'healthy']
    # default weights: more outpatients
    category_weights = [0.20, 0.20, 0.45, 0.15]
    category = category_override if category_override else np.random.choice(categories, p=category_weights)

    # Base defaults
    age = int(np.random.randint(18, 90))
    sex = np.random.choice(["M", "F"])
    diabetes = 0
    hypertension = 0
    bmi = round(np.random.normal(24.0, 4.0), 1)
    bp_sys = int(np.random.normal(120, 12))
    bp_dia = int(np.random.normal(80, 8))
    hr = int(np.random.normal(78, 10))
    temp = round(float(np.random.normal(36.7, 0.5)), 1)
    cough = 0
    fever = 0
    breathless = 0
    wbc = round(float(np.random.normal(7.0, 2.0)), 1)
    crp = round(float(np.random.normal(5.0, 5.0)), 1)
    diagnosis_code = 'D16'

    if category == 'long_inpatient':
        choices = ['D01','D02','D03','D04','D05']
        diagnosis_code = random.choice(choices)
        if diagnosis_code == 'D01':
            age = max(age, int(np.random.normal(72, 8)))
            hypertension = 1
            breathless = 1
            bp_sys = int(np.random.normal(150, 16))
            hr = int(np.random.normal(95, 12))
            wbc = round(np.random.normal(9.5, 2.0), 1)
            crp = round(abs(np.random.normal(25, 15)), 1)
        elif diagnosis_code == 'D02':
            age = max(age, int(np.random.normal(68, 8)))
            breathless = 1
            cough = 1
            hr = int(np.random.normal(105, 12))
            temp = round(abs(np.random.normal(37.5, 0.8)),1)
            wbc = round(np.random.normal(12.0, 3.0), 1)
            crp = round(abs(np.random.normal(45.0, 20.0)), 1)
        elif diagnosis_code == 'D03':
            age = int(np.random.normal(65, 12))
            fever = 1
            temp = round(abs(np.random.normal(38.8, 0.8)),1)
            hr = int(np.random.normal(115, 15))
            bp_sys = int(np.random.normal(95, 12))
            wbc = round(abs(np.random.normal(18.0, 5.0)),1)
            crp = round(abs(np.random.normal(120.0, 40.0)),1)
            diabetes = int(np.random.rand() < 0.4)
        elif diagnosis_code == 'D04':
            age = max(age, int(np.random.normal(75, 7)))
            hypertension = 1
            bp_sys = int(np.random.normal(165, 20))
            hr = int(np.random.normal(85, 10))
            crp = round(abs(np.random.normal(15.0, 10.0)),1)
        elif diagnosis_code == 'D05':
            age = max(age, int(np.random.normal(55, 10)))
            fever = 1
            temp = round(abs(np.random.normal(38.3, 0.6)),1)
            wbc = round(abs(np.random.normal(14.0, 3.0)),1)
            crp = round(abs(np.random.normal(85.0, 30.0)),1)

    elif category == 'short_inpatient':
        choices = ['D06','D07','D08','D09','D10']
        diagnosis_code = random.choice(choices)
        if diagnosis_code == 'D06':
            age = int(abs(np.random.normal(58, 15)))
            cough = 1
            fever = 1
            temp = round(abs(np.random.normal(38.6, 0.7)),1)
            breathless = int(np.random.rand() < 0.6)
            wbc = round(abs(np.random.normal(13.5, 3.0)),1)
            crp = round(abs(np.random.normal(65.0, 30.0)),1)
        elif diagnosis_code == 'D07':
            age = int(abs(np.random.normal(62, 18)))
            fever = 1
            temp = round(abs(np.random.normal(38.4, 0.8)),1)
            hr = int(np.random.normal(98, 12))
            wbc = round(abs(np.random.normal(11.0, 3.0)),1)
            crp = round(abs(np.random.normal(40.0, 25.0)),1)
            diabetes = int(np.random.rand() < 0.5)
        elif diagnosis_code == 'D08':
            age = int(abs(np.random.normal(60, 12)))
            hypertension = 1
            breathless = int(np.random.rand() < 0.4)
            bp_sys = int(np.random.normal(195, 15))
            bp_dia = int(np.random.normal(115, 10))
            hr = int(np.random.normal(95, 12))
        elif diagnosis_code == 'D09':
            age = int(abs(np.random.normal(45, 15)))
            diabetes = 1
            breathless = 1
            temp = round(abs(np.random.normal(37.2, 0.5)),1)
            hr = int(np.random.normal(105, 12))
        elif diagnosis_code == 'D10':
            age = int(abs(np.random.normal(55, 15)))
            fever = 1
            temp = round(abs(np.random.normal(38.2, 0.7)),1)
            wbc = round(abs(np.random.normal(12.0, 3.0)),1)
            crp = round(abs(np.random.normal(55.0, 25.0)),1)
            diabetes = int(np.random.rand() < 0.4)

    elif category == 'outpatient':
        choices = ['D11','D12','D13','D14','D15']
        diagnosis_code = random.choice(choices)
        if diagnosis_code == 'D11':
            age = int(abs(np.random.normal(35, 15)))
            cough = 1
            fever = int(np.random.rand() < 0.7)
            temp = round(abs(np.random.normal(37.8, 0.6)),1) if fever else round(abs(np.random.normal(36.9, 0.3)),1)
            wbc = round(abs(np.random.normal(8.5, 2.0)),1) if fever else round(abs(np.random.normal(7.0,1.0)),1)
        elif diagnosis_code == 'D12':
            age = int(abs(np.random.normal(58, 12)))
            hypertension = 1
            bp_sys = int(np.random.normal(145, 10))
            crp = round(abs(np.random.normal(4.0, 2.0)),1)
        elif diagnosis_code == 'D13':
            age = int(abs(np.random.normal(55, 12)))
            diabetes = 1
            bmi = round(abs(np.random.normal(31.0, 4.0)),1)
            crp = round(abs(np.random.normal(6.0, 3.0)),1)
        elif diagnosis_code == 'D14':
            age = int(abs(np.random.normal(32, 12)))
            fever = int(np.random.rand() < 0.5)
            temp = round(abs(np.random.normal(37.6, 0.7)),1) if fever else round(abs(np.random.normal(36.8,0.3)),1)
            wbc = round(abs(np.random.normal(9.0, 2.0)),1) if fever else round(abs(np.random.normal(7.0,1.0)),1)
            crp = round(abs(np.random.normal(12.0, 8.0)),1) if fever else round(abs(np.random.normal(4.0,1.0)),1)
        elif diagnosis_code == 'D15':
            age = int(abs(np.random.normal(45, 15)))
            bmi = round(abs(np.random.normal(27.0, 4.0)),1)
            hr = int(np.random.normal(80, 8))
            crp = round(abs(np.random.normal(3.0, 2.0)),1)

    elif category == 'healthy':
        diagnosis_code = 'D16'
        age = int(np.random.normal(38, 12))
        bmi = round(np.random.normal(23.5, 3.0),1)
        bp_sys = int(np.random.normal(118, 8))
        bp_dia = int(np.random.normal(76, 6))
        hr = int(np.random.normal(72, 8))
        temp = round(np.random.normal(36.6, 0.3),1)
        wbc = round(np.random.normal(6.5, 1.5),1)
        crp = round(np.random.normal(2.0, 1.5),1)
        cough = 0
        fever = 0
        breathless = 0
        diabetes = 0
        hypertension = 0

    # Calculate severity score using same algorithm as app/train
    severity_score = 0
    if bp_sys > 180 or bp_sys < 90: severity_score += 2
    if bp_dia > 120 or bp_dia < 60: severity_score += 2
    if hr > 120 or hr < 50: severity_score += 2
    if temp > 39.0: severity_score += 2
    if cough or fever or breathless: severity_score += 1
    if cough and fever: severity_score += 1
    if cough and breathless: severity_score += 1
    if fever and breathless: severity_score += 2
    if wbc > 15.0: severity_score += 3
    elif wbc > 11.0: severity_score += 2
    if crp > 50: severity_score += 3
    elif crp > 20: severity_score += 2
    elif crp > 10: severity_score += 1
    if diabetes: severity_score += 1
    if hypertension: severity_score += 1
    if age > 65: severity_score += 1

    # Inpatient assignment and stay logic
    inpatient = 0
    ward_type = 'None'
    stay_days = 0
    if category == 'long_inpatient':
        inpatient = 1
        ward_type = random.choice(['ICU','Cardiac','Neurological','General'])
        if diagnosis_code == 'D01':
            stay_days = int(abs(np.random.normal(10,3)))
        elif diagnosis_code == 'D02':
            stay_days = int(abs(np.random.normal(12,4)))
        elif diagnosis_code == 'D03':
            stay_days = int(abs(np.random.normal(14,5)))
        elif diagnosis_code == 'D04':
            stay_days = int(abs(np.random.normal(18,6)))
        elif diagnosis_code == 'D05':
            stay_days = int(abs(np.random.normal(8,3)))
        stay_days = max(7, min(30, stay_days))
    elif category == 'short_inpatient':
        inpatient = 1
        ward_type = random.choice(['General','ICU','Cardiac','Isolation'])
        if diagnosis_code == 'D06':
            stay_days = int(abs(np.random.normal(3,1)))
        elif diagnosis_code == 'D07':
            stay_days = int(abs(np.random.normal(2,1)))
        elif diagnosis_code == 'D08':
            stay_days = int(abs(np.random.normal(2,1)))
        elif diagnosis_code == 'D09':
            stay_days = int(abs(np.random.normal(3,1)))
        elif diagnosis_code == 'D10':
            stay_days = int(abs(np.random.normal(4,2)))
        stay_days = max(1, min(7, stay_days))
    elif category == 'outpatient':
        inpatient = 0
        if severity_score >= 6 and np.random.rand() < 0.15:
            inpatient = 1
            ward_type = 'General'
            stay_days = 1

    # Clamp numeric values
    bp_sys = int(max(70, min(250, bp_sys)))
    bp_dia = int(max(40, min(150, bp_dia)))
    hr = int(max(30, min(200, hr)))
    temp = float(round(max(30.0, min(42.0, temp)),1))
    wbc = float(round(max(0.0, min(50.0, wbc)),1))
    crp = float(round(max(0.0, min(300.0, abs(crp))),1))
    bmi = float(round(max(12.0, min(60.0, bmi)),1))

    row = {
        "patient_id": f"P{i:06d}",
        "age": int(age),
        "sex": sex,
        "bmi": bmi,
        "blood_pressure_sys": bp_sys,
        "blood_pressure_dia": bp_dia,
        "heart_rate": hr,
        "temperature": temp,
        "symptom_cough": int(cough),
        "symptom_fever": int(fever),
        "symptom_breathless": int(breathless),
        "comorbidity_diabetes": int(diabetes),
        "comorbidity_hypertension": int(hypertension),
        "lab_wbc": wbc,
        "lab_crp": crp,
        "diagnosis_code": diagnosis_code,
        "severity_score": severity_score,
        "inpatient": int(inpatient),
        "ward_type": ward_type,
        "stay_days": int(stay_days)
    }

    # Add engineered flags
    flags = engineered_flags(row)
    row.update(flags)

    # Add medicines list (string) for training medicine recommender
    meds = DIAGNOSIS_MAPPING.get(diagnosis_code, {}).get('medicines', [])
    row['medicines'] = "|".join(meds) if meds else ""

    return row

def generate(n: int, out: str, seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    rows = []
    # Aim for a balanced set across diagnosis groups
    per_group = max(1, n // 16)
    idx = 1
    diag_codes = [
        'D01','D02','D03','D04','D05',
        'D06','D07','D08','D09','D10',
        'D11','D12','D13','D14','D15',
        'D16'
    ]
    # First ensure each diagnosis is represented
    for code in diag_codes:
        # derive the category for forcing to get distribution
        if code in ['D01','D02','D03','D04','D05']:
            category = 'long_inpatient'
        elif code in ['D06','D07','D08','D09','D10']:
            category = 'short_inpatient'
        elif code in ['D11','D12','D13','D14','D15']:
            category = 'outpatient'
        else:
            category = 'healthy'
        for _ in range(per_group):
            row = gen_row(idx, category_override=category)
            # enforce diagnosis code precisely
            row['diagnosis_code'] = code
            # re-calc meds and engineered flags in case
            row['medicines'] = "|".join(DIAGNOSIS_MAPPING.get(code, {}).get('medicines', []))
            flags = engineered_flags(row)
            row.update(flags)
            rows.append(row)
            idx += 1

    # then fill remaining records randomly
    while len(rows) < n:
        row = gen_row(idx)
        rows.append(row)
        idx += 1

    df = pd.DataFrame(rows)

    # Quality clamp and cleaning
    df['age'] = df['age'].clip(0, 120).astype(int)
    df['bmi'] = df['bmi'].clip(12, 60).astype(float)
    df['blood_pressure_sys'] = df['blood_pressure_sys'].clip(70, 250).astype(int)
    df['blood_pressure_dia'] = df['blood_pressure_dia'].clip(40, 150).astype(int)
    df['heart_rate'] = df['heart_rate'].clip(30, 200).astype(int)
    df['temperature'] = df['temperature'].clip(30.0, 42.0).astype(float)
    df['lab_wbc'] = df['lab_wbc'].clip(0.0, 50.0).astype(float)
    df['lab_crp'] = df['lab_crp'].clip(0.0, 300.0).astype(float)
    df['stay_days'] = df['stay_days'].clip(0, 30).astype(int)

    df.to_csv(out, index=False)
    # Print summary
    print(f"Generated {len(df)} records to {out}")
    print("Diagnosis distribution:")
    print(df['diagnosis_code'].value_counts().to_string())
    print(f"Overall inpatient rate: {df['inpatient'].mean():.2%}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2000, help='number of samples')
    parser.add_argument('--out', type=str, default='data/patients_sample.csv', help='output csv path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    print("Generating dataset...")
    df = generate(args.n, args.out, args.seed)
    print("Done.")
