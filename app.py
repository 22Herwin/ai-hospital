from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
import datetime
import time
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import ollama
from icd10_loader import lookup_icd10
import concurrent.futures

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "meditron:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Initialize session state
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None
if 'admission_complete' not in st.session_state:
    st.session_state.admission_complete = False
if 'last_admission_id' not in st.session_state:
    st.session_state.last_admission_id = None
if 'ai_unavailable' not in st.session_state:
    st.session_state['ai_unavailable'] = False

# Safe rerun helper
def safe_rerun():
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()
    else:
        st.session_state['_safe_rerun_toggle'] = not st.session_state.get('_safe_rerun_toggle', False)

st.set_page_config(page_title='AI Hospital Management System', layout='wide')

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
STOCK_CSV = os.path.join(DATA_DIR, 'medicine_stock.csv')
PATIENTS_LOG = os.path.join(DATA_DIR, 'admission_log.csv')
os.makedirs(DATA_DIR, exist_ok=True)

# Disease categories for reference
HOSPITALIZATION_DISEASES = {
    "J18.9": "Pneumonia, unspecified",
    "I21.9": "Acute myocardial infarction", 
    "I63.9": "Cerebral infarction",
    "A41.9": "Sepsis, unspecified",
    "J15.9": "Bacterial pneumonia"
}

OUTPATIENT_DISEASES = {
    "I10": "Essential hypertension",
    "E11.9": "Type 2 diabetes",
    "J06.9": "Acute upper respiratory infection", 
    "K29.70": "Gastritis",
    "M54.50": "Low back pain"
}

HEALTHY_CODE = {
    "Z00.0": "General medical examination"
}

# Stock management functions
@st.cache_resource
def load_stock():
    """Load medicine stock from CSV with fallback initialization"""
    try:
        if os.path.exists(STOCK_CSV):
            df = pd.read_csv(STOCK_CSV)
            df['stock'] = pd.to_numeric(df['stock'], errors='coerce').fillna(0).astype(int)
            return df
        else:
            # Create comprehensive medicine stock
            common_meds = [
                'Amoxicillin 500mg', 'Azithromycin 250mg', 'Paracetamol 500mg',
                'Aspirin 100mg', 'Clopidogrel 75mg', 'Atorvastatin 20mg',
                'Meropenem 1g IV', 'Vancomycin 1g IV', 'IV Fluids',
                'Ceftriaxone 1g IV', 'Oxygen therapy', 'Mannitol IV',
                'Amlodipine 5mg', 'Lisinopril 10mg', 'Metformin 500mg',
                'Omeprazole 20mg', 'Ibuprofen 400mg', 'Chlorpheniramine 4mg'
            ]
            df = pd.DataFrame({'medicine_name': common_meds, 'stock': [25]*len(common_meds)})
            df.to_csv(STOCK_CSV, index=False)
            return df
    except Exception as e:
        st.error(f"Error loading stock: {str(e)}")
        return pd.DataFrame(columns=['medicine_name', 'stock'])

def save_stock(df):
    """Save updated stock to CSV"""
    try:
        df.to_csv(STOCK_CSV, index=False)
        load_stock.clear()
        return True
    except Exception as e:
        st.error(f"Error saving stock: {str(e)}")
        return False

def build_clinical_note(features: dict) -> str:
    """Build comprehensive clinical note for AI analysis"""
    return f"""
PATIENT CLINICAL PRESENTATION:

Demographics:
- Age: {features['age']} years
- Sex: {features.get('sex', 'Not specified')}
- BMI: {features['bmi']:.1f}

Vital Signs:
- Blood Pressure: {features['blood_pressure_sys']}/{features['blood_pressure_dia']} mmHg
- Heart Rate: {features['heart_rate']} bpm
- Temperature: {features['temperature']} °C

Presenting Symptoms:
- Cough: {'Yes' if features['symptom_cough'] else 'No'}
- Fever: {'Yes' if features['symptom_fever'] else 'No'}
- Breathlessness: {'Yes' if features['symptom_breathless'] else 'No'}
- Chest Pain: {'Yes' if features.get('symptom_chest_pain', False) else 'No'}
- Neurological Symptoms: {'Yes' if features.get('symptom_neuro', False) else 'No'}

Comorbidities:
- Diabetes: {'Yes' if features['comorbidity_diabetes'] else 'No'}
- Hypertension: {'Yes' if features['comorbidity_hypertension'] else 'No'}

Laboratory Findings:
- White Blood Cell Count: {features['lab_wbc']} x10^9/L (Normal: 4-11)
- C-Reactive Protein: {features['lab_crp']} mg/L (Normal: <5)

Clinical Severity Assessment:
- Severity Score: {features['severity_score']}/25
"""

def analyze_with_ollama(features: dict) -> dict:
    """
    Analyze patient features using Ollama with Meditron model.
    Runs the model call in a short-lived thread and enforces a timeout so
    the UI doesn't hang indefinitely. On timeout/connection error we mark AI
    as unavailable and return fallback analysis.
    """
    if st.session_state.get('ai_unavailable', False):
        return fallback_analysis(features)

    clinical_note = build_clinical_note(features)
    from ai_engine import analyze_text_with_ollama

    timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT", "12"))

    try:
        start = time.time()
        with st.spinner(f"{OLLAMA_MODEL} analyzing clinical presentation... (timeout {timeout_seconds}s)"):
            # run model call in a worker thread so we can enforce timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(analyze_text_with_ollama, clinical_note)
                ai_result = future.result(timeout=timeout_seconds)

        elapsed = time.time() - start
        # If returned None or malformed, treat as failure and fall back
        if not ai_result or not isinstance(ai_result, dict):
            st.warning("AI returned unexpected result — using fallback rules")
            return fallback_analysis(features)

        # Verify ICD-10 code using local ICD loader
        if ai_result and ai_result.get('icd10_code'):
            try:
                icd_info = lookup_icd10(ai_result['icd10_code'])
                if icd_info and icd_info.get('title'):
                    ai_result['diagnosis_name'] = icd_info.get('title')
                    st.success(f"ICD-10 Code Validated: {ai_result['icd10_code']} (took {elapsed:.1f}s)")
            except Exception as e:
                st.warning(f"ICD-10 verification issue: {str(e)}")

        return ai_result

    except concurrent.futures.TimeoutError:
        st.error(f"Ollama call timed out after {timeout_seconds}s — switching to fallback")
        st.session_state['ai_unavailable'] = True
        return fallback_analysis(features)

    except Exception as e:
        error_msg = str(e)
        st.error(f"Ollama analysis failed: {error_msg}")
        if "connection" in error_msg.lower() or "model" in error_msg.lower():
            st.session_state['ai_unavailable'] = True
            st.sidebar.error("Ollama server/model unavailable — switched to fallback rules.")
        return fallback_analysis(features)

def fallback_analysis(features: dict) -> dict:
    """Fallback rule-based analysis when AI fails"""
    st.warning("Using fallback clinical rules (AI unavailable)")
    
    # Enhanced rule-based diagnosis matching our disease categories
    if features['symptom_cough'] and features['symptom_fever'] and features['lab_wbc'] > 15:
        diagnosis = {'code': 'J18.9', 'name': 'Pneumonia, unspecified', 'meds': ['Amoxicillin 500mg TDS', 'Azithromycin 250mg OD'], 'ward': 'General', 'stay': 5}
    elif features.get('symptom_chest_pain', False) and features['comorbidity_hypertension']:
        diagnosis = {'code': 'I21.9', 'name': 'Acute myocardial infarction', 'meds': ['Aspirin 300mg', 'Clopidogrel 75mg'], 'ward': 'ICU', 'stay': 7}
    elif features.get('symptom_neuro', False):
        diagnosis = {'code': 'I63.9', 'name': 'Cerebral infarction', 'meds': ['Aspirin 100mg', 'Atorvastatin 40mg'], 'ward': 'Neurological', 'stay': 10}
    elif features['symptom_fever'] and features['lab_crp'] > 100:
        diagnosis = {'code': 'A41.9', 'name': 'Sepsis, unspecified', 'meds': ['Meropenem 1g IV', 'IV Fluids'], 'ward': 'ICU', 'stay': 14}
    elif features['comorbidity_hypertension']:
        diagnosis = {'code': 'I10', 'name': 'Essential hypertension', 'meds': ['Amlodipine 5mg OD', 'Lisinopril 10mg OD'], 'ward': 'Outpatient', 'stay': 0}
    elif features['comorbidity_diabetes']:
        diagnosis = {'code': 'E11.9', 'name': 'Type 2 diabetes', 'meds': ['Metformin 500mg BD'], 'ward': 'Outpatient', 'stay': 0}
    else:
        diagnosis = {'code': 'Z00.0', 'name': 'General medical examination', 'meds': ['Routine follow-up'], 'ward': 'Outpatient', 'stay': 0}
    
    is_inpatient = diagnosis['stay'] > 0
    
    return {
        "icd10_code": diagnosis['code'],
        "diagnosis_name": diagnosis['name'],
        "confidence": 0.7,
        "inpatient": is_inpatient,
        "estimated_stay_days": diagnosis['stay'] if is_inpatient else 0,
        "ward_type": diagnosis['ward'],
        "recommended_medicines": diagnosis['meds'],
        "rationale": f"Fallback analysis based on clinical features. Severity: {features['severity_score']}/25"
    }

def calc_severity_score(features: dict) -> int:
    """Calculate clinical severity score"""
    score = 0
    
    # Vital signs abnormalities (more weight)
    if features['blood_pressure_sys'] > 180 or features['blood_pressure_sys'] < 90: score += 4
    elif features['blood_pressure_sys'] > 160 or features['blood_pressure_sys'] < 100: score += 2
    
    if features['blood_pressure_dia'] > 120 or features['blood_pressure_dia'] < 60: score += 3
    if features['heart_rate'] > 120 or features['heart_rate'] < 50: score += 3
    if features['temperature'] > 39.0: score += 3
    elif features['temperature'] > 38.0: score += 2
    
    # Symptoms (weighted by severity)
    if features['symptom_breathless']: score += 4
    if features.get('symptom_chest_pain', False): score += 4
    if features.get('symptom_neuro', False): score += 5
    if features['symptom_fever']: score += 2
    if features['symptom_cough']: score += 1
    
    # Lab values
    if features['lab_wbc'] > 15.0: score += 3
    elif features['lab_wbc'] > 11.0: score += 2
    if features['lab_crp'] > 100: score += 4
    elif features['lab_crp'] > 50: score += 3
    elif features['lab_crp'] > 20: score += 2
    elif features['lab_crp'] > 10: score += 1
    
    # Comorbidities and age
    if features['comorbidity_diabetes']: score += 1
    if features['comorbidity_hypertension']: score += 1
    if features['age'] > 65: score += 2
    elif features['age'] > 50: score += 1
    
    return min(score, 25)

# Sidebar configuration
st.sidebar.header('System Information')

# Disease reference in sidebar
with st.sidebar.expander("Disease Reference Guide", expanded=False):
    st.subheader("Hospitalization Required")
    for code, name in HOSPITALIZATION_DISEASES.items():
        st.write(f"**{code}**: {name}")
    
    st.subheader("Outpatient Management") 
    for code, name in OUTPATIENT_DISEASES.items():
        st.write(f"**{code}**: {name}")
    
    st.subheader("Healthy")
    st.write(f"**Z00.0**: General medical examination")

if st.session_state.get('ai_unavailable', False):
    st.sidebar.error(f"{OLLAMA_MODEL} unavailable — using fallback rules")
    if st.sidebar.button("Retry AI Connection"):
        st.session_state['ai_unavailable'] = False
        safe_rerun()
else:
    st.sidebar.success(f"Ollama Connected")
    st.sidebar.info(f"Model: {OLLAMA_MODEL}")

st.sidebar.info("""
**Clinical Decision Support**
- Specific disease categorization
- ICD-10 code validation
- Hospitalization criteria
- Evidence-based treatment
""")

# Main content
st.title('AI Hospital Management System')
st.markdown("### Specific Disease Diagnosis & Patient Management")

# Enhanced patient input form
with st.form('patient_form'):
    st.subheader("Patient Clinical Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pid = st.text_input('Patient ID', value=f"P{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
        age = st.number_input('Age', min_value=0, max_value=120, value=45)
        sex = st.selectbox('Biological Sex', ['M', 'F', 'Other'])
        bmi = st.number_input('BMI', value=24.0, step=0.1, min_value=10.0, max_value=60.0)
    
    with col2:
        st.subheader("Vital Signs")
        bp_sys = st.number_input('Systolic BP (mmHg)', value=120, min_value=70, max_value=250)
        bp_dia = st.number_input('Diastolic BP (mmHg)', value=80, min_value=40, max_value=150)
        hr = st.number_input('Heart Rate (bpm)', value=78, min_value=30, max_value=200)
        temp = st.number_input('Temperature (°C)', value=36.7, step=0.1, min_value=30.0, max_value=42.0)
    
    with col3:
        st.subheader("Clinical Indicators")
        cough = st.checkbox('Cough')
        fever = st.checkbox('Fever (>38°C)')
        breathless = st.checkbox('Breathlessness')
        chest_pain = st.checkbox('Chest Pain')
        neuro = st.checkbox('Neurological Symptoms')
        diabetes = st.checkbox('Diabetes')
        hypertension = st.checkbox('Hypertension')
        wbc = st.number_input('WBC Count (10^9/L)', value=7.0, step=0.1, min_value=0.0, max_value=50.0)
        crp = st.number_input('CRP Level (mg/L)', value=5.0, step=0.1, min_value=0.0, max_value=300.0)
    
    submitted = st.form_submit_button(f'Diagnose with {OLLAMA_MODEL}', type='primary')

if submitted:
    # Calculate severity score
    features = {
        'pid': pid, 'age': age, 'sex': sex, 'bmi': bmi,
        'blood_pressure_sys': bp_sys, 'blood_pressure_dia': bp_dia,
        'heart_rate': hr, 'temperature': temp,
        'symptom_cough': cough, 'symptom_fever': fever, 'symptom_breathless': breathless,
        'symptom_chest_pain': chest_pain, 'symptom_neuro': neuro,
        'comorbidity_diabetes': diabetes, 'comorbidity_hypertension': hypertension,
        'lab_wbc': wbc, 'lab_crp': crp
    }
    severity_score = calc_severity_score(features)
    features['severity_score'] = severity_score
    
    # Store patient data
    st.session_state.current_patient = {
        'pid': pid,
        'features': features,
        'severity_score': severity_score,
        'timestamp': datetime.datetime.now()
    }
    
    # Analyze with AI
    ai_result = analyze_with_ollama(features)
    
    if ai_result:
        st.session_state.current_patient['ai_result'] = ai_result
    
    st.session_state.admission_complete = False
    safe_rerun()

# Display analysis results
if st.session_state.current_patient and not st.session_state.admission_complete:
    patient = st.session_state.current_patient
    features = patient['features']
    ai_result = patient.get('ai_result', {})
    
    st.subheader("Clinical Analysis Results")
    
    # Severity and basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = patient['severity_score']
        st.metric("Clinical Severity Score", f"{score}/25", 
                 delta="Critical" if score >= 15 else 
                       "High Risk" if score >= 10 else 
                       "Moderate Risk" if score >= 5 else "Low Risk",
                 delta_color="inverse")
    
    with col2:
        if ai_result:
            icd_code = ai_result.get('icd10_code', 'Unknown')
            # Color code based on hospitalization need
            if icd_code in HOSPITALIZATION_DISEASES:
                st.error("HOSPITALIZATION REQUIRED")
            elif icd_code in OUTPATIENT_DISEASES:
                st.warning("OUTPATIENT MANAGEMENT")
            else:
                st.success("HEALTHY / ROUTINE CARE")
    
    with col3:
        if ai_result:
            confidence = ai_result.get('confidence', 0.7) * 100
            st.metric("AI Confidence", f"{confidence:.1f}%")

    # Diagnosis details
    st.markdown("---")
    st.subheader("Diagnosis & Treatment Plan")
    
    if ai_result:
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**ICD-10 Code:** {ai_result.get('icd10_code', 'Unknown')}")
            st.success(f"**Diagnosis:** {ai_result.get('diagnosis_name', 'Unknown')}")
            
            inpatient_status = ai_result.get('inpatient', False)
            if inpatient_status:
                st.error(f"**Hospitalization:** REQUIRED")
                st.warning(f"**Ward Type:** {ai_result.get('ward_type', 'General')}")
                st.info(f"**Estimated Stay:** {ai_result.get('estimated_stay_days', 3)} days")
            else:
                st.success(f"**Hospitalization:** Not Required")
                st.info(f"**Care Setting:** {ai_result.get('ward_type', 'Outpatient')}")
        
        with col2:
            st.subheader("Recommended Medications")
            meds = ai_result.get('recommended_medicines', [])
            if meds:
                for i, med in enumerate(meds, 1):
                    st.write(f"{i}. {med}")
            else:
                st.info("No specific medications recommended")
    
    # Clinical rationale
    with st.expander("Clinical Reasoning", expanded=True):
        if ai_result:
            st.write(ai_result.get('rationale', 'No detailed rationale provided'))
        else:
            st.write("Analysis in progress...")
    
    # Admission workflow for inpatients
    if ai_result and ai_result.get('inpatient'):
        st.markdown("---")
        st.subheader("Admission Workflow")
        
        stock_df = load_stock()
        recommended_meds = ai_result.get('recommended_medicines', [])
        
        # Filter available medicines
        available_meds = stock_df[
            stock_df['medicine_name'].apply(lambda x: any(med in x for med in recommended_meds)) & 
            (stock_df['stock'] > 0)
        ]
        
        if available_meds.empty:
            st.warning("No recommended medicines in stock. Please replenish inventory.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                selected_med = st.selectbox(
                    'Select Medication to Assign', 
                    available_meds['medicine_name'].tolist(),
                    help="Choose from available in-stock medications"
                )
            
            with col2:
                current_stock = available_meds[available_meds['medicine_name'] == selected_med]['stock'].values[0]
                qty = st.number_input(
                    'Quantity to Assign', 
                    min_value=1, 
                    max_value=int(current_stock), 
                    value=1,
                    help=f"Available stock: {current_stock}"
                )
            
            if st.button('CONFIRM ADMISSION', type='primary', use_container_width=True):
                # Create admission record
                admission_data = {
                    'patient_id': patient['pid'],
                    'admit_time': patient['timestamp'].isoformat(),
                    'ward_type': ai_result.get('ward_type', 'General'),
                    'estimated_days': ai_result.get('estimated_stay_days', 3),
                    'med_used': selected_med,
                    'qty': int(qty),
                    'diagnosis_code': ai_result.get('icd10_code', 'Unknown'),
                    'diagnosis_name': ai_result.get('diagnosis_name', 'Unknown'),
                    'severity_score': patient['severity_score']
                }
                
                # Save admission to log
                os.makedirs(DATA_DIR, exist_ok=True)
                admission_df = pd.DataFrame([admission_data])
                if os.path.exists(PATIENTS_LOG):
                    admission_df.to_csv(PATIENTS_LOG, mode='a', header=False, index=False)
                else:
                    admission_df.to_csv(PATIENTS_LOG, index=False)
                
                # Update stock
                stock_df.loc[stock_df['medicine_name'] == selected_med, 'stock'] -= qty
                save_stock(stock_df)
                
                st.success(f"Patient {patient['pid']} admitted to {admission_data['ward_type']} ward")
                st.session_state.last_admission_id = patient['pid']
                st.session_state.admission_complete = True
                safe_rerun()

# Inventory management
st.markdown("---")
st.subheader("Medicine Inventory Management")

stock_df = load_stock()
st.dataframe(stock_df)

col1, col2 = st.columns(2)
with col1:
    if st.button('Refresh Inventory'):
        load_stock.clear()
        safe_rerun()
with col2:
    if st.button('Replenish All Stock (+20 units)'):
        stock_df['stock'] = stock_df['stock'] + 20
        save_stock(stock_df)
        st.success("All stock replenished by 20 units")

# Admission history
st.markdown("---")
st.subheader("Recent Admissions")
if os.path.exists(PATIENTS_LOG):
    try:
        log_df = pd.read_csv(PATIENTS_LOG)
        st.dataframe(log_df.tail(5))
    except Exception as e:
        st.error(f"Error loading admission log: {str(e)}")
else:
    st.info("No admission history available yet")

st.markdown("---")
st.caption(f"AI Hospital Management System • Powered by {OLLAMA_MODEL} • ICD-10 Validated • © 2025")

def deterministic_classifier(features: dict) -> dict:
    """
    Stage 1 (deterministic): decide ICD-10 code, name, ward and base meds
    using rule tables. This MUST be authoritative for diagnosis/code.
    """
    # Keep rules aligned with fallback_analysis but return a compact base
    if features['symptom_cough'] and features['symptom_fever'] and features['lab_wbc'] > 15:
        base = {'icd10_code': 'J18.9', 'diagnosis_name': 'Pneumonia, unspecified',
                'inpatient': True, 'estimated_stay_days': 5, 'ward_type': 'General',
                'recommended_medicines': ['Amoxicillin 500mg TDS', 'Azithromycin 250mg OD']}
    elif features.get('symptom_chest_pain', False) and features['comorbidity_hypertension']:
        base = {'icd10_code': 'I21.9', 'diagnosis_name': 'Acute myocardial infarction',
                'inpatient': True, 'estimated_stay_days': 7, 'ward_type': 'ICU',
                'recommended_medicines': ['Aspirin 300mg', 'Clopidogrel 75mg']}
    elif features.get('symptom_neuro', False):
        base = {'icd10_code': 'I63.9', 'diagnosis_name': 'Cerebral infarction',
                'inpatient': True, 'estimated_stay_days': 10, 'ward_type': 'Neurological',
                'recommended_medicines': ['Aspirin 100mg', 'Atorvastatin 40mg']}
    elif features['symptom_fever'] and features['lab_crp'] > 100:
        base = {'icd10_code': 'A41.9', 'diagnosis_name': 'Sepsis, unspecified',
                'inpatient': True, 'estimated_stay_days': 14, 'ward_type': 'ICU',
                'recommended_medicines': ['Meropenem 1g IV', 'IV Fluids']}
    elif features['comorbidity_hypertension']:
        base = {'icd10_code': 'I10', 'diagnosis_name': 'Essential hypertension',
                'inpatient': False, 'estimated_stay_days': 0, 'ward_type': 'Outpatient',
                'recommended_medicines': ['Amlodipine 5mg OD', 'Lisinopril 10mg OD']}
    elif features['comorbidity_diabetes']:
        base = {'icd10_code': 'E11.9', 'diagnosis_name': 'Type 2 diabetes',
                'inpatient': False, 'estimated_stay_days': 0, 'ward_type': 'Outpatient',
                'recommended_medicines': ['Metformin 500mg BD']}
    else:
        base = {'icd10_code': 'Z00.0', 'diagnosis_name': 'General medical examination',
                'inpatient': False, 'estimated_stay_days': 0, 'ward_type': 'Outpatient',
                'recommended_medicines': ['Routine follow-up']}

    # ensure ICD lookup/title normalization if available
    try:
        icd_info = lookup_icd10(base['icd10_code'])
        if icd_info and icd_info.get('title'):
            base['diagnosis_name'] = icd_info.get('title')
    except Exception:
        pass

    return base

def format_with_ollama(base_result: dict, features: dict, timeout_seconds: Optional[int] = None) -> dict:
    """
    Stage 2 (Ollama): provide human-readable explanation, expand meds, set confidence,
    but DO NOT change the authoritative icd10_code or diagnosis_name from base_result.
    If Ollama fails or times out, return a merged result using base_result.
    """
    from ai_engine import analyze_text_with_ollama

    if timeout_seconds is None:
        timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT", "12"))

    # Build explicit prompt: do not change diagnosis/code, only expand and output JSON
    prompt_text = (
        "You are a clinical assistant. BASE_DIAGNOSIS is authoritative and must NOT be changed.\n"
        "BASE: code={code}, name={name}\n"
        "PATIENT FEATURES: {features}\n\n"
        "Task: Expand clinical rationale, suggest medications (align with provided ward/level), "
        "and output a JSON object with keys: "
        "'icd10_code' (must equal BASE code), 'diagnosis_name' (must equal BASE name), "
        "'confidence' (0-1), 'inpatient' (bool), 'estimated_stay_days' (int), 'ward_type' (string), "
        "'recommended_medicines' (list of strings), 'rationale' (string).\n"
        "Return only a parsable JSON object (no surrounding text)."
    ).format(code=base_result['icd10_code'], name=base_result['diagnosis_name'], features=json.dumps(features))

    try:
        start = time.time()
        with st.spinner(f"{OLLAMA_MODEL} formatting explanation... (timeout {timeout_seconds}s)"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(analyze_text_with_ollama, prompt_text)
                ai_out = future.result(timeout=timeout_seconds)

        # If AI returned a string try to parse JSON, otherwise accept dict
        if isinstance(ai_out, str):
            try:
                ai_parsed = json.loads(ai_out)
            except Exception:
                ai_parsed = {}
        elif isinstance(ai_out, dict):
            ai_parsed = ai_out
        else:
            ai_parsed = {}

        # If parsed dict empty -> fallback to deterministic
        if not ai_parsed:
            st.warning("Ollama returned unexpected format; using deterministic result.")
            merged = {
                "icd10_code": base_result['icd10_code'],
                "diagnosis_name": base_result['diagnosis_name'],
                "confidence": 0.75,
                "inpatient": base_result['inpatient'],
                "estimated_stay_days": base_result['estimated_stay_days'],
                "ward_type": base_result['ward_type'],
                "recommended_medicines": base_result.get('recommended_medicines', []),
                "rationale": f"Deterministic diagnosis used. Severity: {features.get('severity_score',0)}/25"
            }
            return merged

        # Safe extraction with defaults
        try:
            confidence = float(ai_parsed.get('confidence', 0.75))
        except Exception:
            confidence = 0.75

        est_raw = ai_parsed.get('estimated_stay_days')
        if est_raw is None:
            est_days = base_result['estimated_stay_days']
        else:
            try:
                est_days = int(est_raw)
            except Exception:
                est_days = base_result['estimated_stay_days']

        ward_type = ai_parsed.get('ward_type', base_result['ward_type'])
        inpatient_flag = bool(ai_parsed.get('inpatient', base_result['inpatient']))

        meds = ai_parsed.get('recommended_medicines')
        if not isinstance(meds, list):
            meds = base_result.get('recommended_medicines', [])

        rationale = ai_parsed.get('rationale') or ai_parsed.get('explanation') or f"Deterministic diagnosis used. Severity: {features.get('severity_score',0)}/25"

        merged = {
            "icd10_code": base_result['icd10_code'],
            "diagnosis_name": base_result['diagnosis_name'],
            "confidence": confidence,
            "inpatient": inpatient_flag,
            "estimated_stay_days": est_days,
            "ward_type": ward_type,
            "recommended_medicines": meds,
            "rationale": rationale
        }

        # small safety: if meds empty, keep deterministic meds
        if not merged['recommended_medicines']:
            merged['recommended_medicines'] = base_result.get('recommended_medicines', [])

        return merged

    except concurrent.futures.TimeoutError:
        st.error(f"Ollama formatting timed out after {timeout_seconds}s — using deterministic result.")
        st.session_state['ai_unavailable'] = True
        return {
            "icd10_code": base_result['icd10_code'],
            "diagnosis_name": base_result['diagnosis_name'],
            "confidence": 0.7,
            "inpatient": base_result['inpatient'],
            "estimated_stay_days": base_result['estimated_stay_days'],
            "ward_type": base_result['ward_type'],
            "recommended_medicines": base_result.get('recommended_medicines', []),
            "rationale": f"Deterministic diagnosis used after Ollama timeout. Severity: {features.get('severity_score',0)}/25"
        }

    except Exception as e:
        st.warning(f"Ollama formatting failed: {str(e)} — using deterministic result.")
        return {
            "icd10_code": base_result['icd10_code'],
            "diagnosis_name": base_result['diagnosis_name'],
            "confidence": 0.7,
            "inpatient": base_result['inpatient'],
            "estimated_stay_days": base_result['estimated_stay_days'],
            "ward_type": base_result['ward_type'],
            "recommended_medicines": base_result.get('recommended_medicines', []),
            "rationale": f"Deterministic diagnosis used after Ollama error. Severity: {features.get('severity_score',0)}/25"
        }