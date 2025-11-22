from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import time
from typing import Any, Dict, List, Tuple
import subprocess, sys

# load .env
BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Model & data dirs
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STOCK_CSV = os.path.join(DATA_DIR, "medicine_stock.csv")
PATIENTS_LOG = os.path.join(DATA_DIR, "admission_log.csv")

# DIAGNOSIS_MAPPING (keep synchronized with generator/train)
DIAGNOSIS_MAPPING = {
    'D01': {'name':'Heart Failure Exacerbation','description':'Acute worsening of chronic heart failure','medicines':['Furosemide 40mg IV','Carvedilol 6.25mg','Spironolactone 25mg']},
    'D02': {'name':'COPD Exacerbation with Respiratory Failure','description':'Severe COPD flare','medicines':['Methylprednisolone 40mg IV','Levofloxacin 750mg','Albuterol Nebulizer']},
    'D03': {'name':'Sepsis with Multi-organ Dysfunction','description':'Systemic infection with organ failure','medicines':['Vancomycin 15mg/kg IV','Piperacillin-Tazobactam 4.5g IV','Norepinephrine infusion']},
    'D04': {'name':'Stroke with Rehabilitation Needs','description':'Cerebrovascular accident requiring rehab','medicines':['Aspirin 325mg','Atorvastatin 80mg','Clopidogrel 75mg']},
    'D05': {'name':'Pancreatitis with Complications','description':'Severe pancreatitis with complications','medicines':['Pantoprazole 40mg IV','Morphine PCA','Octreotide infusion']},
    'D06': {'name':'Pneumonia (Community Acquired)','description':'Bacterial lung infection','medicines':['Ceftriaxone 1g IV','Azithromycin 500mg IV','Oxygen therapy']},
    'D07': {'name':'UTI with Sepsis','description':'Urinary tract infection with systemic response','medicines':['Ceftriaxone 2g IV','Gentamicin 5mg/kg IV','IV Fluids']},
    'D08': {'name':'Hypertensive Crisis','description':'Severe hypertension requiring urgent BP lowering','medicines':['Labetalol infusion','Nicardipine drip','Hydralazine 10mg IV']},
    'D09': {'name':'Diabetic Ketoacidosis (Resolved)','description':'Metabolic emergency treated with insulin and fluids','medicines':['Insulin infusion','Potassium replacement','IV Fluids']},
    'D10': {'name':'Cellulitis with Systemic Symptoms','description':'Skin infection with systemic features','medicines':['Vancomycin 15mg/kg IV','Clindamycin 600mg IV','Wound care']},
    'D11': {'name':'Upper Respiratory Infection','description':'Usually viral, supportive care','medicines':['Paracetamol 500mg','Dextromethorphan 15mg','Saline nasal spray']},
    'D12': {'name':'Hypertension Management','description':'Chronic blood pressure management','medicines':['Amlodipine 5mg','Lisinopril 10mg','Hydrochlorothiazide 12.5mg']},
    'D13': {'name':'Type 2 Diabetes Follow-up','description':'Diabetes monitoring and medication adjustments','medicines':['Metformin 1000mg','Sitagliptin 100mg','Glipizide 5mg']},
    'D14': {'name':'Gastroenteritis','description':'Stomach/intestine inflammation, usually self-limited','medicines':['Ondansetron 4mg','Loperamide 2mg','Oral Rehydration Salts']},
    'D15': {'name':'Back Pain','description':'Musculoskeletal back pain','medicines':['Ibuprofen 400mg','Cyclobenzaprine 5mg','Acetaminophen 650mg']},
    'D16': {'name':'Well Visit','description':'Routine check-up','medicines':[]}
}

# safe rerun helper
def safe_rerun():
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()
    else:
        st.session_state['_safe_rerun_toggle'] = not st.session_state.get('_safe_rerun_toggle', False)

st.set_page_config(page_title='AI Hospital Management System', layout='wide')

# session state init
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None
if 'admission_complete' not in st.session_state:
    st.session_state.admission_complete = False
if 'last_admission_id' not in st.session_state:
    st.session_state.last_admission_id = None

# utility to load stock CSV
@st.cache_resource
def load_stock():
    try:
        df = pd.read_csv(STOCK_CSV)
        df['stock'] = pd.to_numeric(df['stock'], errors='coerce').fillna(0).astype(int)
        return df
    except Exception:
        # create default demo stock if not present
        meds = []
        for v in DIAGNOSIS_MAPPING.values():
            meds.extend(v.get('medicines', []))
        meds = sorted(list(set(meds)))
        df = pd.DataFrame({'medicine_name': meds, 'stock': [10]*len(meds)})
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(STOCK_CSV, index=False)
        df['stock'] = df['stock'].astype(int)
        return df

def save_stock(df: pd.DataFrame):
    try:
        df.to_csv(STOCK_CSV, index=False)
        # clear cached loader so updates are visible next render
        load_stock.clear()
        return True
    except Exception as e:
        st.error(f"Error saving stock: {e}")
        return False

# fallback diagnosis rule-based (keeps app functional when model missing)
class FallbackDiagnosisModel:
    def predict(self, X):
        row = X.iloc[0]
        if row.get('symptom_cough',0) and row.get('symptom_fever',0) and row.get('lab_wbc',0) > 11:
            return ['D06']
        if row.get('comorbidity_hypertension',0) and row.get('age',0) > 60:
            return ['D01']
        if row.get('comorbidity_diabetes',0) and row.get('lab_crp',0) > 20:
            return ['D09']
        if row.get('symptom_breathless',0) and row.get('temperature',0) > 38:
            return ['D02']
        return ['D16']
    def predict_proba(self, X) -> np.ndarray:
        # return shape (n_samples, n_classes_like) — simple single-column confidence
        n = len(X)
        return np.ones((n, 1), dtype=float) * 0.75

def load_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        'diagnosis': None,
        'inpatient': None,
        'ward': None,
        'stay': None,
        'medicine': None  # medicine model will be a dict {'pipeline':..., 'mlb':...}
    }
    # Try to load
    try:
        models['diagnosis'] = joblib.load(os.path.join(MODELS_DIR, 'diagnosis_model.pkl'))
        st.sidebar.success("Diagnosis model loaded")
    except Exception:
        models['diagnosis'] = FallbackDiagnosisModel()
        st.sidebar.warning("Diagnosis model not found, using fallback rules")

    for name in ['inpatient','ward','stay']:
        try:
            models[name] = joblib.load(os.path.join(MODELS_DIR, f'{name}_model.pkl'))
            st.sidebar.success(f"{name.capitalize()} model loaded")
        except Exception:
            models[name] = None
            st.sidebar.info(f"{name.capitalize()} model not found")

    try:
        med_obj = joblib.load(os.path.join(MODELS_DIR, 'medicine_model.pkl'))
        # medicine_model.pkl stored as dict {'pipeline':..., 'mlb':...}
        if isinstance(med_obj, dict) and 'pipeline' in med_obj and 'mlb' in med_obj:
            models['medicine'] = med_obj
            st.sidebar.success("Medicine recommender loaded")
        else:
            st.sidebar.warning("Medicine model file content unexpected")
    except Exception:
        models['medicine'] = None
        st.sidebar.info("Medicine recommender not found; will use DIAGNOSIS_MAPPING fallback")

    return models

# explicit typing here helps the static checker later when we call predict / named_steps / etc.
models: Dict[str, Any] = load_models()

# helper: engineered feature generator (must match train/generate)
def engineered_flags_from_features(features: dict) -> dict:
    flags = {}
    flags['is_respiratory'] = int((features.get('symptom_cough',0) == 1) or (features.get('symptom_breathless',0) == 1))
    flags['is_infection'] = int((features.get('symptom_fever',0) == 1) and (features.get('lab_crp',0) > 20 or features.get('lab_wbc',0) > 11))
    flags['is_cardiac'] = int((features.get('comorbidity_hypertension',0) == 1) or (features.get('blood_pressure_sys',0) > 160) or (features.get('heart_rate',0) > 110))
    flags['is_metabolic'] = int(features.get('comorbidity_diabetes',0) == 1)
    flags['is_neuro'] = 0
    flags['is_gi'] = int(features.get('diagnosis_code','') in ['D05','D14'])
    flags['age_over_65'] = int(features.get('age',0) > 65)
    return flags

# Calculate severity score (same logic used in dataset and training)
def calc_severity_score(features: dict) -> int:
    score = 0
    if features['blood_pressure_sys'] > 180 or features['blood_pressure_sys'] < 90: score += 2
    if features['blood_pressure_dia'] > 120 or features['blood_pressure_dia'] < 60: score += 2
    if features['heart_rate'] > 120 or features['heart_rate'] < 50: score += 2
    if features['temperature'] > 39.0: score += 2
    if features['symptom_cough'] or features['symptom_fever'] or features['symptom_breathless']: score += 1
    if features['symptom_cough'] and features['symptom_fever']: score += 1
    if features['symptom_cough'] and features['symptom_breathless']: score += 1
    if features['symptom_fever'] and features['symptom_breathless']: score += 2
    if features['lab_wbc'] > 15.0: score += 3
    elif features['lab_wbc'] > 11.0: score += 2
    if features['lab_crp'] > 50: score += 3
    elif features['lab_crp'] > 20: score += 2
    elif features['lab_crp'] > 10: score += 1
    if features['comorbidity_diabetes']: score += 1
    if features['comorbidity_hypertension']: score += 1
    if features['age'] > 65: score += 1
    return score

# Sidebar controls
st.sidebar.header("System Controls")
if st.sidebar.button("Refresh Models"):
    # clear cached resources used by app so models/stock reload on rerun
    load_stock.clear()
    safe_rerun()
if st.sidebar.button("Generate Sample Dataset (1k)"):
    with st.spinner("Generating dataset..."):
        subprocess.run([sys.executable, os.path.join(BASE_DIR,'generate_dataset.py'), '--n','1000','--out',os.path.join(DATA_DIR,'patients_sample.csv')])
    st.sidebar.success("Dataset generated")
if st.sidebar.button("Train Models"):
    with st.spinner("Training models..."):
        subprocess.run([sys.executable, os.path.join(BASE_DIR,'train_models.py'),'--data',os.path.join(DATA_DIR,'patients_sample.csv'),'--out_dir',MODELS_DIR])
    st.sidebar.success("Training triggered - refresh models after complete")

st.title("Hospital Management System")
st.markdown("Patient admission, diagnosis, ward/stay predictions, and medicine recommendations.")

# Patient input form
with st.form("patient_form"):
    st.subheader("Patient Clinical Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        pid = st.text_input("Patient ID", value=f"P{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        sex = st.selectbox("Sex", ['M','F','Other'])
        bmi = st.number_input("BMI", value=24.0, step=0.1, min_value=10.0, max_value=60.0)
    with col2:
        st.subheader("Vitals")
        bp_sys = st.number_input("Systolic BP (mmHg)", value=120, min_value=70, max_value=250)
        bp_dia = st.number_input("Diastolic BP (mmHg)", value=80, min_value=40, max_value=150)
        hr = st.number_input("Heart Rate (bpm)", value=78, min_value=30, max_value=200)
        temp = st.number_input("Temperature (°C)", value=36.7, step=0.1, min_value=30.0, max_value=42.0)
    with col3:
        st.subheader("Clinical Indicators & Labs")
        cough = st.checkbox("Cough")
        fever = st.checkbox("Fever (>38°C)")
        breathless = st.checkbox("Breathlessness")
        diabetes = st.checkbox("Diabetes")
        hypertension = st.checkbox("Hypertension")
        wbc = st.number_input("WBC Count (10^9/L)", value=7.0, step=0.1, min_value=0.0, max_value=50.0)
        crp = st.number_input("CRP Level (mg/L)", value=5.0, step=0.1, min_value=0.0, max_value=300.0)
    submitted = st.form_submit_button("Analyze Patient")

if submitted:
    # Build feature dict
    features = {
        'age': int(age), 'bmi': float(bmi),
        'blood_pressure_sys': int(bp_sys), 'blood_pressure_dia': int(bp_dia),
        'heart_rate': int(hr), 'temperature': float(temp),
        'symptom_cough': int(cough), 'symptom_fever': int(fever), 'symptom_breathless': int(breathless),
        'comorbidity_diabetes': int(diabetes), 'comorbidity_hypertension': int(hypertension),
        'lab_wbc': float(wbc), 'lab_crp': float(crp)
    }
    severity_score = calc_severity_score(features)
    features['severity_score'] = severity_score
    # engineered flags
    flags = engineered_flags_from_features(features)
    features.update(flags)

    st.session_state.current_patient = {
        'pid': pid,
        'features': features,
        'severity_score': severity_score,
        'timestamp': datetime.datetime.now()
    }
    st.session_state.admission_complete = False
    safe_rerun()

# Display results when available
if st.session_state.current_patient and not st.session_state.admission_complete:
    patient = st.session_state.current_patient
    st.subheader("Clinical Analysis Results")
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("Features used:")
        df_disp = pd.DataFrame.from_dict(patient['features'], orient='index', columns=['value'])
        st.dataframe(df_disp)
    with col2:
        st.metric("Severity Score", patient['severity_score'], delta="High" if patient['severity_score']>=10 else "Moderate" if patient['severity_score']>=5 else "Low")

    # Prepare feature DataFrame for models
    feat = patient['features']
    X = pd.DataFrame([{
        'age': feat['age'],'bmi': feat['bmi'],'blood_pressure_sys': feat['blood_pressure_sys'],
        'blood_pressure_dia': feat['blood_pressure_dia'],'heart_rate': feat['heart_rate'],'temperature': feat['temperature'],
        'symptom_cough': feat['symptom_cough'],'symptom_fever': feat['symptom_fever'],'symptom_breathless': feat['symptom_breathless'],
        'comorbidity_diabetes': feat['comorbidity_diabetes'],'comorbidity_hypertension': feat['comorbidity_hypertension'],
        'lab_wbc': feat['lab_wbc'],'lab_crp': feat['lab_crp'],'severity_score': feat['severity_score'],
        'is_respiratory': feat.get('is_respiratory',0),'is_infection': feat.get('is_infection',0),
        'is_cardiac': feat.get('is_cardiac',0),'is_metabolic': feat.get('is_metabolic',0),
        'is_neuro': feat.get('is_neuro',0),'is_gi': feat.get('is_gi',0),'age_over_65': feat.get('age_over_65',0)
    }])

    # DIAGNOSIS PREDICTION
    st.markdown("---")
    st.subheader("AI Diagnosis")
    try:
        diag_pred = models['diagnosis'].predict(X)[0]
        # sanitize diag_pred
        if isinstance(diag_pred, (list, np.ndarray)):
            diag_code = str(diag_pred[0])
        else:
            diag_code = str(diag_pred)
        # ensure format Dxx
        if diag_code.isdigit():
            diag_code = f"D{int(diag_code):02d}"
        if diag_code not in DIAGNOSIS_MAPPING:
            # fallback selection by severity bucket
            if patient['severity_score'] >= 12:
                diag_code = np.random.choice(['D01','D02','D03','D04','D05'])
            elif patient['severity_score'] >= 6:
                diag_code = np.random.choice(['D06','D07','D08','D09','D10'])
            else:
                diag_code = np.random.choice(['D11','D12','D13','D14','D15','D16'])
        diagnosis_data = DIAGNOSIS_MAPPING.get(diag_code, {'name':'Unknown','description':'','medicines':[]})
        st.success(f"{diag_code} - {diagnosis_data['name']}")
        st.caption(diagnosis_data['description'])
        # confidence if available
        try:
            proba = models['diagnosis'].predict_proba(X)[0]
            # proba for multiclass: we can attempt to find index for predicted label
            # but sklearn pipeline returns class labels in clf.classes_
            if hasattr(models['diagnosis'].named_steps['clf'], 'classes_'):
                classes = models['diagnosis'].named_steps['clf'].classes_
                idx = np.where(classes == diag_code)[0]
                if len(idx) > 0:
                    conf = float(proba[idx[0]])*100
                    st.caption(f"Diagnosis confidence: {conf:.1f}%")
        except Exception:
            pass
        # persist
        st.session_state.current_patient['diagnosis_code'] = diag_code
        st.session_state.current_patient['diagnosis_data'] = diagnosis_data
    except Exception as e:
        st.error(f"Diagnosis prediction failed: {e}")
        # fallback mapping
        diag_code = 'D16'
        st.session_state.current_patient['diagnosis_code'] = diag_code
        st.session_state.current_patient['diagnosis_data'] = DIAGNOSIS_MAPPING.get(diag_code,{ 'name':'Unknown','medicines':[] })

    # INPATIENT / OUTPATIENT DECISION
    st.markdown("---")
    st.subheader("Admission Recommendation")
    inpatient_recommendation = False
    inpatient_confidence = None
    try:
        # Add diagnosis_code into features if model expects it
        X_in = X.copy()
        X_in['diagnosis_code'] = st.session_state.current_patient.get('diagnosis_code', 'D16')
        if models['inpatient'] is not None:
            pred_in = models['inpatient'].predict(X_in)[0]
            pred_int = int(pred_in[0]) if isinstance(pred_in, (list,tuple,np.ndarray)) else int(pred_in)
            inpatient_recommendation = (pred_int == 1)
            # try to get probability
            try:
                proba = models['inpatient'].predict_proba(X_in)
                # if binary, proba shape (n,2)
                if proba.shape[1] == 2:
                    inpatient_confidence = float(proba[0,1]*100)
                else:
                    inpatient_confidence = float(proba[0,0]*100)
            except Exception:
                inpatient_confidence = None
        else:
            # fallback by severity
            inpatient_recommendation = patient['severity_score'] >= 10
    except Exception as e:
        st.warning(f"Inpatient prediction error: {e}")
        inpatient_recommendation = patient['severity_score'] >= 10

    if inpatient_recommendation:
        if inpatient_confidence:
            st.success(f"HOSPITALIZATION RECOMMENDED (Confidence: {inpatient_confidence:.1f}%)")
        else:
            st.success("HOSPITALIZATION RECOMMENDED")
        # Ward & stay predictions
        ward = 'General'
        stay_days = 3
        try:
            X_ward = X.copy()
            X_ward['diagnosis_code'] = st.session_state.current_patient['diagnosis_code']
            if models['ward'] is not None:
                ward_pred = models['ward'].predict(X_ward)[0]
                ward = str(ward_pred[0]) if isinstance(ward_pred, (list,tuple,np.ndarray)) else str(ward_pred)
            if models['stay'] is not None:
                stay_pred = models['stay'].predict(X_ward)[0]
                stay_days = max(1, int(round(float(stay_pred))))
        except Exception as e:
            st.warning(f"Ward/stay prediction failed: {e}")
        st.info(f"Recommended Ward: {ward}")
        st.info(f"Estimated Stay: {stay_days} days")
    else:
        st.info("Outpatient care recommended")
        if inpatient_confidence is not None:
            st.caption(f"Model inpatient confidence: {inpatient_confidence:.1f}%")

    # MEDICINE RECOMMENDATION
    st.markdown("---")
    st.subheader("Medication Recommendation")
    diagnosis_data = st.session_state.current_patient.get('diagnosis_data', {})
    recommended_from_map = diagnosis_data.get('medicines', [])
    ml_recommended = []
    # Use medicine ML recommender if present
    try:
        if models['medicine'] is not None:
            med_pipeline = models['medicine']['pipeline']
            mlb = models['medicine']['mlb']
            X_med = X.copy()
            X_med['diagnosis_code'] = st.session_state.current_patient.get('diagnosis_code','D16')
            y_pred = med_pipeline.predict(X_med)
            # y_pred is binary matrix
            labels = mlb.inverse_transform(y_pred)
            if len(labels) > 0 and len(labels[0]) > 0:
                ml_recommended = list(labels[0])
    except Exception as e:
        st.warning(f"Medicine recommender failed: {e}")
        ml_recommended = []

    # Final recommended meds: prefer ML recommender when non-empty, else DIAGNOSIS_MAPPING
    final_recommend = ml_recommended if len(ml_recommended) > 0 else recommended_from_map
    if final_recommend:
        st.info("Recommended medicines (automated):")
        st.markdown("```\n" + "\n".join([f"- {m}" for m in final_recommend]) + "\n```")
    else:
        st.warning("No automated medicine recommendation available. Please consult clinician.")

    # Inventory and admission (if decided to admit)
    if inpatient_recommendation:
        st.markdown("---")
        st.subheader("Medication Assignment & Admission")
        stock_df = load_stock()
        # show recommended meds and available ones
        available = stock_df[stock_df['medicine_name'].isin(final_recommend) & (stock_df['stock'] > 0)]
        if available.empty:
            st.warning("No recommended medicines are currently in stock.")
        else:
            med_choice = st.selectbox("Select medication to assign", available['medicine_name'].tolist())
            sel_row = available[available['medicine_name'] == med_choice].iloc[0]
            current_stock = int(sel_row['stock'])
            qty = st.number_input("Quantity", min_value=1, max_value=current_stock, value=1)
            if st.button("Confirm Admission & Assign Medication"):
                # Process admission record (local storage fallback)
                admission = {
                    'patient_id': patient['pid'],
                    'admit_time': patient['timestamp'].isoformat(),
                    'ward_type': ward,
                    'estimated_days': stay_days,
                    'med_used': med_choice,
                    'qty': int(qty),
                    'diagnosis_code': st.session_state.current_patient.get('diagnosis_code','D16'),
                    'severity_score': patient['severity_score']
                }
                # Save admission to CSV
                os.makedirs(DATA_DIR, exist_ok=True)
                admission_df = pd.DataFrame([admission])
                if os.path.exists(PATIENTS_LOG):
                    admission_df.to_csv(PATIENTS_LOG, mode='a', header=False, index=False)
                else:
                    admission_df.to_csv(PATIENTS_LOG, index=False)
                # decrement stock
                mask = stock_df['medicine_name'] == med_choice
                stock_df.loc[mask, 'stock'] = stock_df.loc[mask, 'stock'] - int(qty)
                save_stock(stock_df)
                st.success("Admission recorded and stock updated.")
                st.session_state.last_admission_id = f"LOCAL-{int(time.time())}"
                st.session_state.admission_complete = True
                safe_rerun()

# Inventory management and recent admissions
st.markdown("---")
st.subheader("Medicine Inventory")
stock_df = load_stock()
st.dataframe(stock_df)
col1, col2 = st.columns(2)
with col1:
    if st.button("Replenish Low Stock (+10 each)"):
        low = stock_df[stock_df['stock'] < 5]
        for idx in low.index:
            # safely coerce to numeric (handles strings/None) and treat NaN as 0
            val = pd.to_numeric(stock_df.at[idx, 'stock'], errors='coerce')
            if np.isnan(val):
                val = 0
            stock_df.at[idx, 'stock'] = int(val) + 10
        save_stock(stock_df)
        st.success("Low stock items replenished.")
with col2:
    if st.button("Full Replenish (+10 each)"):
        # ensure numeric dtype and use Series.add to satisfy type-checkers
        stock_df['stock'] = stock_df['stock'].astype(int).add(10)
        save_stock(stock_df)
        st.success("All items replenished.")

st.markdown("---")
st.subheader("Recent Admissions (local log)")
if os.path.exists(PATIENTS_LOG):
    try:
        log_df = pd.read_csv(PATIENTS_LOG)
        st.dataframe(log_df.tail(10))
    except Exception as e:
        st.error(f"Failed to read admission log: {e}")
else:
    st.info("No local admission log found.")

st.markdown("---")
st.caption("Hospital Management System Created by K-7 @2025")
