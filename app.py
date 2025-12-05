from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
import datetime
import time
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import requests
from icd10_loader import lookup_icd10
import concurrent.futures
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Chutes AI configuration
CHUTES_MODEL = os.getenv("CHUTES_MODEL", "unsloth/gemma-3-12b-it")
CHUTES_API_URL = os.getenv("CHUTES_API_URL", "https://llm.chutes.ai/v1/chat/completions")
CHUTES_API_TOKEN = os.getenv("CHUTES_API_TOKEN")

# Initialize session state
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None
if 'admission_complete' not in st.session_state:
    st.session_state.admission_complete = False
if 'last_admission_id' not in st.session_state:
    st.session_state.last_admission_id = None
if 'ai_unavailable' not in st.session_state:
    st.session_state['ai_unavailable'] = False

# If token missing, mark AI as unavailable (app keeps working with deterministic fallback)
if not CHUTES_API_TOKEN:
    st.session_state['ai_unavailable'] = True

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
HOSPITALS_CSV = os.path.join(DATA_DIR, 'hospitals.csv')  # <-- new
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
            # Only return if non-empty
            if not df.empty:
                return df
        
        # Always create default stock if missing or empty
        common_meds = [
            'Amoxicillin 500mg', 'Azithromycin 250mg', 'Paracetamol 500mg',
            'Aspirin 100mg', 'Clopidogrel 75mg', 'Atorvastatin 20mg',
            'Meropenem 1g IV', 'Vancomycin 1g IV', 'IV Fluids',
            'Ceftriaxone 1g IV', 'Oxygen therapy', 'Mannitol IV',
            'Amlodipine 5mg', 'Lisinopril 10mg', 'Metformin 500mg',
            'Omeprazole 20mg', 'Ibuprofen 400mg', 'Chlorpheniramine 4mg'
        ]
        df = pd.DataFrame({'medicine_name': common_meds, 'stock': [50]*len(common_meds)})  # Increased stock to 50
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(STOCK_CSV, index=False)
        st.info("Medicine stock initialized with default inventory")
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

@st.cache_resource
def load_hospitals():
    """Load hospitals with ward capacity and occupancy; create defaults if missing."""
    try:
        if os.path.exists(HOSPITALS_CSV):
            df = pd.read_csv(HOSPITALS_CSV)
            # Ensure columns exist and are numeric; if missing, initialize with zeros
            if 'total_beds' in df.columns:
                df['total_beds'] = pd.to_numeric(df['total_beds'], errors='coerce').fillna(0).astype(int)
            else:
                df['total_beds'] = 0
            if 'occupied_beds' in df.columns:
                df['occupied_beds'] = pd.to_numeric(df['occupied_beds'], errors='coerce').fillna(0).astype(int)
            else:
                df['occupied_beds'] = 0
            return df
        else:
            # Create default hospitals with ward capacity
            default = [
                {'hospital_id': 'H-01', 'hospital_name': 'Central Medical Hospital', 'ward_type': 'General', 'total_beds': 20, 'occupied_beds': 15},
                {'hospital_id': 'H-02', 'hospital_name': 'Central Medical Hospital', 'ward_type': 'ICU', 'total_beds': 6, 'occupied_beds': 4},
                {'hospital_id': 'H-03', 'hospital_name': 'Central Medical Hospital', 'ward_type': 'Neurological', 'total_beds': 8, 'occupied_beds': 3},
                {'hospital_id': 'H-04', 'hospital_name': 'City General Hospital', 'ward_type': 'General', 'total_beds': 25, 'occupied_beds': 8},
                {'hospital_id': 'H-05', 'hospital_name': 'City General Hospital', 'ward_type': 'ICU', 'total_beds': 8, 'occupied_beds': 2},
                {'hospital_id': 'H-06', 'hospital_name': 'City General Hospital', 'ward_type': 'Neurological', 'total_beds': 10, 'occupied_beds': 5},
            ]
            df = pd.DataFrame(default)
            df.to_csv(HOSPITALS_CSV, index=False)
            return df
    except Exception as e:
        st.error(f"Error loading hospitals: {e}")
        return pd.DataFrame(columns=['hospital_id', 'hospital_name', 'ward_type', 'total_beds', 'occupied_beds'])

def save_hospitals(df: pd.DataFrame):
    """Persist hospitals CSV and clear cache."""
    try:
        df.to_csv(HOSPITALS_CSV, index=False)
        load_hospitals.clear()
        return True
    except Exception as e:
        st.error(f"Error saving hospitals: {e}")
        return False

def find_available_hospital(requested_ward: str, hospitals_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Find a hospital ward row with available beds for the requested ward type.

    Returns a dict with hospital fields and computed 'available_beds', or None if none found.
    """
    try:
        # Ensure numeric types
        df = hospitals_df.copy()
        if 'total_beds' not in df.columns or 'occupied_beds' not in df.columns:
            return None
        df['total_beds'] = pd.to_numeric(df['total_beds'], errors='coerce').fillna(0).astype(int)
        df['occupied_beds'] = pd.to_numeric(df['occupied_beds'], errors='coerce').fillna(0).astype(int)

        # Filter by requested ward type (case-insensitive)
        candidates = df[df['ward_type'].str.lower() == str(requested_ward).lower()].copy()
        if candidates.empty:
            return None

        # Compute available beds
        candidates['available_beds'] = candidates['total_beds'] - candidates['occupied_beds']
        candidates = candidates[candidates['available_beds'] > 0]

        if candidates.empty:
            return None

        # Prefer hospital with most available beds, tie-breaker: lowest occupancy %
        candidates['occupancy_pct'] = candidates.apply(
            lambda r: (r['occupied_beds'] / r['total_beds']) if r['total_beds'] > 0 else 1.0,
            axis=1
        )
        candidates = candidates.sort_values(by=['available_beds', 'occupancy_pct'], ascending=[False, True])

        row = candidates.iloc[0]
        return {
            'hospital_id': row.get('hospital_id'),
            'hospital_name': row.get('hospital_name'),
            'ward_type': row.get('ward_type'),
            'total_beds': int(row.get('total_beds', 0)),
            'occupied_beds': int(row.get('occupied_beds', 0)),
            'available_beds': int(row.get('available_beds', 0))
        }
    except Exception:
        return None

def check_ward_capacity_and_alert(requested_ward: str, hospitals_df: pd.DataFrame) -> Tuple[bool, str]:
    """Check aggregate capacity for a ward type and return (has_capacity, message)."""
    try:
        df = hospitals_df.copy()
        # Ensure numeric columns exist
        if 'total_beds' not in df.columns or 'occupied_beds' not in df.columns:
            return False, "Hospital data missing bed information."

        df['total_beds'] = pd.to_numeric(df['total_beds'], errors='coerce').fillna(0).astype(int)
        df['occupied_beds'] = pd.to_numeric(df['occupied_beds'], errors='coerce').fillna(0).astype(int)

        matching = df[df['ward_type'].str.lower() == str(requested_ward).lower()]
        if matching.empty:
            return False, f"No wards of type '{requested_ward}' found in hospital list."

        total_beds = int(matching['total_beds'].sum())
        total_occupied = int(matching['occupied_beds'].sum())
        total_available = total_beds - total_occupied
        occupancy_pct = int((total_occupied / total_beds) * 100) if total_beds > 0 else 100

        if total_available > 0:
            msg = f"{total_available} bed(s) available across {matching['hospital_name'].nunique()} hospital(s) for '{requested_ward}' ({occupancy_pct}% occupied)."
            return True, msg
        else:
            # No beds available
            if occupancy_pct >= 95:
                note = "CRITICAL - no beds, consider overflow/transfer."
            elif occupancy_pct >= 85:
                note = "WARNING - capacity critically low."
            else:
                note = "No beds available at the moment."
            msg = f"No available beds for '{requested_ward}' ({occupancy_pct}% occupied). {note}"
            return False, msg
    except Exception as e:
        return False, f"Capacity check error: {str(e)}"


def build_clinical_note(features: dict) -> str:
    """Build comprehensive clinical note for AI analysis"""
    return f"""
PATIENT CLINICAL PRESENTATION:

Demographics:
- Age: {features['age']} years
- Sex: {'Unknown' if features.get('sex') not in ['M', 'F'] else features['sex']}
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

def analyze_with_chutes(features: dict) -> dict:
    """
    Analyze patient features using Chutes AI with retry, logging and a configurable timeout.
    """
    if st.session_state.get('ai_unavailable', False):
        return fallback_analysis(features)

    clinical_note = build_clinical_note(features)
    from ai_engine import analyze_text_with_chutes

    # configurable timeout (set CHUTES_TIMEOUT in .env). Default increased to 30s.
    timeout_seconds = int(os.getenv("CHUTES_TIMEOUT", "30"))
    raw_log = os.path.join(DATA_DIR, "chutes_raw_responses.log")
    attempts = 2

    for attempt in range(1, attempts + 1):
        try:
            start = time.time()
            with st.spinner(f"{CHUTES_MODEL} analyzing clinical presentation... (timeout {timeout_seconds}s) [attempt {attempt}/{attempts}]"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(analyze_text_with_chutes, clinical_note)
                    ai_result = future.result(timeout=timeout_seconds)

            elapsed = time.time() - start

            # Log raw response for debugging (safe: truncated)
            try:
                with open(raw_log, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now().isoformat()} | attempt={attempt} | elapsed={elapsed:.2f}s | raw={repr(ai_result)[:4000]}\n")
            except Exception:
                pass

            # Validate result
            if not ai_result or not isinstance(ai_result, dict) or not ai_result.get("icd10_code"):
                st.warning("AI returned unexpected or incomplete result — using fallback rules")
                return fallback_analysis(features)

            # Verify ICD-10 code using local ICD loader
            try:
                icd_info = lookup_icd10(ai_result['icd10_code'])
                if icd_info and icd_info.get('title'):
                    ai_result['diagnosis_name'] = icd_info.get('title')
                    st.success(f"ICD-10 Code Validated: {ai_result['icd10_code']} (took {elapsed:.1f}s)")
            except Exception as e:
                st.warning(f"ICD-10 verification issue: {str(e)}")

            return ai_result

        except concurrent.futures.TimeoutError:
            # timeout: retry once, then mark unavailable
            msg = f"Chutes AI call timed out after {timeout_seconds}s (attempt {attempt}/{attempts})"
            st.warning(msg)
            with open(raw_log, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now().isoformat()} | TIMEOUT | attempt={attempt}\n")
            if attempt < attempts:
                time.sleep(1)  # short backoff then retry
                continue
            st.error(msg + " — switching to fallback")
            st.session_state['ai_unavailable'] = True
            return fallback_analysis(features)

        except Exception as e:
            # other errors: log and fallback
            err = str(e)
            try:
                with open(raw_log, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now().isoformat()} | ERROR | attempt={attempt} | err={err[:2000]}\n")
            except Exception:
                pass
            if "token" in err.lower() or "connection" in err.lower():
                st.session_state['ai_unavailable'] = True
                st.sidebar.error("Chutes AI token/connection issue — switched to fallback rules.")
            # on first attempt try again, otherwise fallback
            if attempt < attempts:
                time.sleep(1)
                continue
            return fallback_analysis(features)

    # Ensure a dict is always returned on every code path (static checkers require this)
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
    st.sidebar.error(f"{CHUTES_MODEL} unavailable — using fallback rules")
    if st.sidebar.button("Retry AI Connection"):
        # Try to re-check availability by reloading environment (user expected to set CHUTES_API_TOKEN)
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
        if os.getenv("CHUTES_API_TOKEN"):
            st.session_state['ai_unavailable'] = False
        safe_rerun()
else:
    st.sidebar.success(f"Chutes AI Connected")
    st.sidebar.info(f"Model: {CHUTES_MODEL}")

# <-- NEW: Hospital Occupancy Display
with st.sidebar.expander("Hospital Occupancy Status", expanded=True):
    hospitals_df = load_hospitals()
    unique_hospitals = hospitals_df['hospital_name'].unique()
    
    for hospital_name in unique_hospitals:
        st.write(f"**{hospital_name}**")
        hospital_wards = hospitals_df[hospitals_df['hospital_name'] == hospital_name]
        
        for _, row in hospital_wards.iterrows():
            available = int(row['total_beds']) - int(row['occupied_beds'])
            occupancy_pct = int((int(row['occupied_beds']) / int(row['total_beds'])) * 100)
            status_color = "🔴" if occupancy_pct >= 90 else "🟡" if occupancy_pct >= 70 else "🟢"
            st.caption(f"{status_color} {row['ward_type']}: {available}/{int(row['total_beds'])} beds ({occupancy_pct}%)")
        
        st.divider()
# <-- END NEW

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
        def generate_sequential_patient_id():
            """Generate sequential patient ID from log file"""
            import os
            counter_file = os.path.join(DATA_DIR, '.patient_counter')
            
            try:
                if os.path.exists(counter_file):
                    with open(counter_file, 'r') as f:
                        count = int(f.read().strip())
                else:
                    count = 0
                
                count += 1
                with open(counter_file, 'w') as f:
                    f.write(str(count))
                
                return f"P{count:06d}"  # P000001, P000002, etc
            except Exception:
                return f"P{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        auto_pid = generate_sequential_patient_id()
        
        st.text_input(
            'Patient ID (Auto-Generated)',
            value=auto_pid,
            disabled=True,
            help="Sequential patient ID"
        )
        
        pid = auto_pid
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

    submitted = st.form_submit_button(f'Diagnose with {CHUTES_MODEL}', type='primary')

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
    ai_result = analyze_with_chutes(features)

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
        hospitals_df = load_hospitals()
        recommended_meds = ai_result.get('recommended_medicines', [])
        requested_ward = ai_result.get('ward_type', 'General')

        # Find available hospital for requested ward type
        available_hospital = find_available_hospital(requested_ward, hospitals_df)

        if available_hospital is None:
            st.error(f"No beds available in any hospital for {requested_ward} ward")
            st.info("All hospitals are at full capacity for the requested ward type. Consider:")
            st.write("- Placing patient on waiting list")
            st.write("- Transferring to another hospital type")
            st.write("- Delaying non-urgent admission")
        else:
            # Show selected hospital info
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"Hospital Assignment")
                st.write(f"**Hospital:** {available_hospital['hospital_name']}")
                st.write(f"**Ward Type:** {available_hospital['ward_type']}")
                st.write(f"**Available Beds:** {available_hospital['available_beds']}/{available_hospital['total_beds']}")
            
            with col2:
                st.info(f"**Occupancy:** {int((available_hospital['occupied_beds'] / available_hospital['total_beds']) * 100)}%")

            # Filter available medicines - IMPROVED
            available_meds = stock_df[
                stock_df['medicine_name'].apply(lambda x: any(med.lower() in x.lower() for med in recommended_meds)) &
                (stock_df['stock'] > 0)
            ]

            if available_meds.empty:
                # If no exact match, show all available meds
                st.warning("Exact recommended medicines not in stock. Showing all available medicines:")
                available_meds = stock_df[stock_df['stock'] > 0]
                
                if available_meds.empty:
                    st.error("No medicines in stock. Please replenish inventory first.")
                    st.stop()

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
            
            # NEW: Admission Date Planning
            st.write("**Planned Admission Schedule**")
            col_date1, col_date2, col_date3 = st.columns(3)
            
            with col_date1:
                admission_date = st.date_input(
                    "Admission Date",
                    value=datetime.date.today(),
                    min_value=datetime.date.today(),
                    help="Date when patient will be admitted"
                )
            
            with col_date2:
                admission_time = st.time_input(
                    "Admission Time",
                    value=datetime.time(hour=9, minute=0),
                    help="Time of admission"
                )
            
            with col_date3:
                discharge_days = st.number_input(
                    "Estimated Length of Stay (days)",
                    min_value=1,
                    max_value=90,
                    value=ai_result.get('estimated_stay_days', 3),
                    help="How many days will patient be admitted?"
                )
            
            # Calculate estimated discharge date
            admission_datetime = datetime.datetime.combine(admission_date, admission_time)
            estimated_discharge = admission_date + datetime.timedelta(days=int(discharge_days))
            
            col_summary1, col_summary2 = st.columns(2)
            with col_summary1:
                st.info(f"📅 **Admission:** {admission_date.strftime('%Y-%m-%d (%A)')} at {admission_time.strftime('%H:%M')}")
            
            with col_summary2:
                st.info(f"📅 **Est. Discharge:** {estimated_discharge.strftime('%Y-%m-%d (%A)')} ({int(discharge_days)} days)")

            if st.button('CONFIRM ADMISSION', type='primary', width='stretch'):
                # Check current capacity before admission
                has_capacity, capacity_msg = check_ward_capacity_and_alert(requested_ward, hospitals_df)
                
                st.info(capacity_msg)
                
                if not has_capacity:
                    st.error(f"Cannot admit: {capacity_msg}")
                    st.stop()
                
                # Create admission record with hospital info AND DATE PLANNING
                admission_data = {
                    'patient_id': patient['pid'],
                    'admit_time': admission_datetime.isoformat(),  # Use planned admission time
                    'planned_admission_date': admission_date.isoformat(),
                    'planned_admission_time': admission_time.isoformat(),
                    'estimated_discharge_date': estimated_discharge.isoformat(),
                    'length_of_stay_days': int(discharge_days),
                    'hospital_id': available_hospital['hospital_id'],
                    'hospital_name': available_hospital['hospital_name'],
                    'ward_type': available_hospital['ward_type'],
                    'estimated_days': ai_result.get('estimated_stay_days', 3),
                    'med_used': selected_med,
                    'qty': int(qty),
                    'diagnosis_code': ai_result.get('icd10_code', 'Unknown'),
                    'diagnosis_name': ai_result.get('diagnosis_name', 'Unknown'),
                    'severity_score': patient['severity_score']
                }

                # Update hospital occupancy
                hospitals_df.loc[hospitals_df['hospital_id'] == available_hospital['hospital_id'], 'occupied_beds'] = \
                    int(available_hospital['occupied_beds']) + 1
                save_hospitals(hospitals_df)

                # SKIP LOCAL CSV - SAVE DIRECTLY TO SUPABASE ONLY
                supabase_success = False
                try:
                    from supabase_client import insert_admission
                    result = insert_admission(admission_data)
                    if result:
                        supabase_success = True
                        st.success(f"✅ Patient {patient['pid']} admitted to {admission_data['hospital_name']} ({admission_data['ward_type']} Ward)")
                    else:
                        st.error("Failed to save to Supabase")
                        st.stop()
                except ImportError:
                    st.error("Supabase client not available. Install: pip install supabase")
                    st.stop()
                except Exception as e:
                    st.error(f"Supabase error: {type(e).__name__}: {str(e)}")
                    st.stop()

                # Update stock
                stock_df.loc[stock_df['medicine_name'] == selected_med, 'stock'] -= qty
                save_stock(stock_df)

                # Show success messages
                st.success(f"Patient {patient['pid']} admitted")
                st.info(f"Medicine assigned: {selected_med} x{qty}")
                st.info(f"Admission: {admission_date.strftime('%Y-%m-%d')} → Discharge: {estimated_discharge.strftime('%Y-%m-%d')} ({int(discharge_days)} days)")
                st.info(f"Data stored in Supabase")
                
                # Calculate occupancy_pct AFTER updating hospitals
                updated_hospitals_df = load_hospitals()
                hospital_data = updated_hospitals_df[updated_hospitals_df['hospital_id'] == available_hospital['hospital_id']]
                if not hospital_data.empty:
                    updated_total = int(hospital_data.iloc[0]['total_beds'])
                    updated_occupied = int(hospital_data.iloc[0]['occupied_beds'])
                    occupancy_pct = int((updated_occupied / updated_total) * 100) if updated_total > 0 else 0
                    
                    if occupancy_pct >= 85:
                        st.warning(f"Ward now at {occupancy_pct}% capacity")
                
                st.session_state.last_admission_id = patient['pid']
                st.session_state.admission_complete = True
                
                time.sleep(2)
                
                # Clear caches to force reload
                load_hospitals.clear()
                load_stock.clear()
                st.rerun()

# Inventory management
st.markdown("---")
st.subheader("Medicine Inventory Management")

stock_df = load_stock()

col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Refresh Inventory'):
        load_stock.clear()
        safe_rerun()

with col2:
    if st.button('Replenish All Stock (+50 units)'):
        stock_df['stock'] = stock_df['stock'] + 50
        save_stock(stock_df)
        st.success("All stock replenished by 50 units")
        load_stock.clear()
        safe_rerun()

with col3:
    if st.button('Reset to Default (50 units each)'):
        common_meds = [
            'Amoxicillin 500mg', 'Azithromycin 250mg', 'Paracetamol 500mg',
            'Aspirin 100mg', 'Clopidogrel 75mg', 'Atorvastatin 20mg',
            'Meropenem 1g IV', 'Vancomycin 1g IV', 'IV Fluids',
            'Ceftriaxone 1g IV', 'Oxygen therapy', 'Mannitol IV',
            'Amlodipine 5mg', 'Lisinopril 10mg', 'Metformin 500mg',
            'Omeprazole 20mg', 'Ibuprofen 400mg', 'Chlorpheniramine 4mg'
        ]
        stock_df = pd.DataFrame({'medicine_name': common_meds, 'stock': [50]*len(common_meds)})
        save_stock(stock_df)
        st.success("Stock reset to defaults")
        load_stock.clear()
        safe_rerun()

st.dataframe(stock_df, width='stretch')

# <-- NEW: Ward Capacity Monitor (Graphical)
st.markdown("---")
st.subheader("Ward Capacity Monitor & Auto-Routing Status")

hospitals_df = load_hospitals()

# Prepare data for visualizations
ward_summary = []
for ward_type in ['General', 'ICU', 'Neurological']:
    df = hospitals_df.copy()
    df['total_beds'] = pd.to_numeric(df['total_beds'], errors='coerce').fillna(0).astype(int)
    df['occupied_beds'] = pd.to_numeric(df['occupied_beds'], errors='coerce').fillna(0).astype(int)
    
    matching = df[df['ward_type'].str.lower() == ward_type.lower()]
    if matching.empty:
        continue
    
    total_beds = int(matching['total_beds'].sum())
    total_occupied = int(matching['occupied_beds'].sum())
    total_available = total_beds - total_occupied
    occupancy_pct = int((total_occupied / total_beds) * 100) if total_beds > 0 else 0
    
    ward_summary.append({
        'ward_type': ward_type,
        'total_beds': total_beds,
        'occupied_beds': total_occupied,
        'available_beds': total_available,
        'occupancy_pct': occupancy_pct
    })

# Create tabs for different views
tab1, tab2 = st.tabs(["Overall Summary", "Per-Hospital Breakdown"])

# Tab 1: Overall Summary Charts
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart: Occupancy by Ward Type
        fig_bar = go.Figure()
        for item in ward_summary:
            fig_bar.add_trace(go.Bar(
                x=[item['ward_type']],
                y=[item['occupancy_pct']],
                name=item['ward_type'],
                text=f"{item['occupancy_pct']}%",
                textposition='outside',
                marker=dict(
                    color='red' if item['occupancy_pct'] >= 95 else
                           'orange' if item['occupancy_pct'] >= 85 else
                           'yellow' if item['occupancy_pct'] >= 70 else 'green'
                )
            ))
        
        fig_bar.update_layout(
            title="Ward Occupancy % by Type",
            yaxis_title="Occupancy %",
            xaxis_title="Ward Type",
            showlegend=False,
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Stacked bar: Occupied vs Available beds
        fig_stack = go.Figure()
        ward_types = [item['ward_type'] for item in ward_summary]
        occupied = [item['occupied_beds'] for item in ward_summary]
        available = [item['available_beds'] for item in ward_summary]
        
        fig_stack.add_trace(go.Bar(
            x=ward_types,
            y=occupied,
            name='Occupied Beds',
            marker_color='indianred'
        ))
        fig_stack.add_trace(go.Bar(
            x=ward_types,
            y=available,
            name='Available Beds',
            marker_color='lightgreen'
        ))
        
        fig_stack.update_layout(
            barmode='stack',
            title="Bed Availability by Ward Type",
            yaxis_title="Number of Beds",
            xaxis_title="Ward Type",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_stack, use_container_width=True)

# Tab 2: Per-Hospital Breakdown
with tab2:
    # Create breakdown data
    hospital_breakdown = []
    for _, row in hospitals_df.iterrows():
        hospital_breakdown.append({
            'Hospital': row['hospital_name'],
            'Ward': row['ward_type'],
            'Total Beds': int(row['total_beds']),
            'Occupied': int(row['occupied_beds']),
            'Available': int(row['total_beds']) - int(row['occupied_beds']),
            'Occupancy %': int((int(row['occupied_beds']) / int(row['total_beds']) * 100) if int(row['total_beds']) > 0 else 0)
        })
    
    breakdown_df = pd.DataFrame(hospital_breakdown)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Grouped bar chart by hospital and ward
        fig_grouped = px.bar(
            breakdown_df,
            x='Ward',
            y='Occupancy %',
            color='Hospital',
            barmode='group',
            title="Occupancy % by Hospital & Ward",
            height=400
        )
        fig_grouped.update_yaxes(range=[0, 105])
        st.plotly_chart(fig_grouped, use_container_width=True)
    
    with col2:
        # Heatmap: Hospital vs Ward Occupancy
        pivot_df = breakdown_df.pivot_table(
            index='Hospital',
            columns='Ward',
            values='Occupancy %',
            aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
        )
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn_r',
            text=pivot_df.values,
            texttemplate='%{text:.0f}%',
            textfont={"size": 12},
            colorbar=dict(title="Occupancy %")
        ))
        
        fig_heatmap.update_layout(
            title="Hospital & Ward Occupancy Heatmap",
            xaxis_title="Ward Type",
            yaxis_title="Hospital",
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Table view
    st.write("**Detailed Breakdown:**")
    st.dataframe(breakdown_df, width='stretch', hide_index=True)

# Summary statistics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

total_beds_all = sum([item['total_beds'] for item in ward_summary])
total_occupied_all = sum([item['occupied_beds'] for item in ward_summary])
total_available_all = sum([item['available_beds'] for item in ward_summary])
avg_occupancy = int((total_occupied_all / total_beds_all * 100) if total_beds_all > 0 else 0)

with col1:
    st.metric("Total Beds", total_beds_all)

with col2:
    st.metric("Occupied Beds", total_occupied_all)

with col3:
    st.metric("Available Beds", total_available_all)

with col4:
    st.metric("Average Occupancy", f"{avg_occupancy}%")

# <-- END NEW

# Hospital management
st.markdown("---")
st.subheader("Hospital Bed Management")

hospitals_df = load_hospitals()
st.dataframe(hospitals_df, width='stretch')

col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Refresh Hospital Status'):
        load_hospitals.clear()
        safe_rerun()

with col2:
    if st.button('Reset All Occupancy (Clear Beds)'):
        hospitals_df['occupied_beds'] = 0
        save_hospitals(hospitals_df)
        st.success("All beds cleared")
        safe_rerun()

with col3:
    if st.button('Simulate Admissions (+2 per ward)'):
        hospitals_df['occupied_beds'] = (hospitals_df['occupied_beds'] + 2).clip(upper=hospitals_df['total_beds'])
        save_hospitals(hospitals_df)
        st.info("Simulated +2 admissions per ward")
        safe_rerun()

# View Supabase admissions
st.markdown("---")
st.subheader("Supabase Admissions Sync")

col1, col2 = st.columns(2)

with col1:
    if st.button('Fetch from Supabase'):
        try:
            from supabase_client import get_all_admissions
            admissions = get_all_admissions()
            if admissions:
                admissions_df = pd.DataFrame(admissions)
                st.dataframe(admissions_df, width='stretch')
                st.success(f"Synced {len(admissions)} admissions found in Supabase")
            else:
                st.info("No admissions in Supabase yet")
        except Exception as e:
            st.error(f"Error fetching from Supabase: {str(e)}")


# View admission timeline
st.markdown("---")
st.subheader("Patient Admission Timeline")

try:
    from supabase_client import get_all_admissions
    import pandas as pd
    
    admissions = get_all_admissions()
    
    if admissions:
        admissions_df = pd.DataFrame(admissions)
        
        # Convert admit_time to datetime
        admissions_df['admit_time'] = pd.to_datetime(admissions_df['admit_time'], format='ISO8601', utc=True)
        admissions_df = admissions_df.sort_values('admit_time', ascending=False)  # Most recent first
        
        # Extract date components for filtering
        admissions_df['date'] = admissions_df['admit_time'].dt.date
        admissions_df['month'] = admissions_df['admit_time'].dt.to_period('M')
        admissions_df['year'] = admissions_df['admit_time'].dt.year
        
        # Create tabs for different timeline views
        timeline_tab1, timeline_tab2, timeline_tab3 = st.tabs(["Recent Admissions", "Analytics", "Hospital Breakdown"])
        
        with timeline_tab1:
            # FILTERS SECTION
            st.write("**Filter Options**")
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                filter_type = st.radio(
                    "Filter by:",
                    ["All", "Date Range", "Specific Month", "Specific Date"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
            
            filtered_df = admissions_df.copy()
            
            with col_filter2:
                if filter_type == "Date Range":
                    date_range = st.date_input(
                        "Select date range",
                        value=(admissions_df['date'].min(), admissions_df['date'].max()),
                        min_value=admissions_df['date'].min(),
                        max_value=admissions_df['date'].max(),
                        key="date_range"
                    )
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start_date, end_date = date_range
                        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
                    elif isinstance(date_range, tuple):
                        st.warning("Please select both start and end date")
                
                elif filter_type == "Specific Month":
                    available_months = sorted(admissions_df['month'].unique(), reverse=True)
                    selected_month = st.selectbox(
                        "Select month",
                        available_months,
                        format_func=lambda x: str(x),
                        key="month_filter"
                    )
                    filtered_df = filtered_df[filtered_df['month'] == selected_month]
                
                elif filter_type == "Specific Date":
                    available_dates = sorted(admissions_df['date'].unique(), reverse=True)
                    selected_date = st.selectbox(
                        "Select date",
                        available_dates,
                        format_func=lambda x: x.strftime('%Y-%m-%d'),
                        key="date_filter"
                    )
                    filtered_df = filtered_df[filtered_df['date'] == selected_date]
            
            with col_filter3:
                filter_ward = st.multiselect(
                    "Filter by Ward Type",
                    admissions_df['ward_type'].unique().tolist(),
                    default=admissions_df['ward_type'].unique().tolist(),
                    key="ward_filter"
                )
                if filter_ward:
                    filtered_df = filtered_df[filtered_df['ward_type'].isin(filter_ward)]
            
            st.divider()
            
            # Display filtered results summary
            st.write(f"**Showing {len(filtered_df)} admission(s)** out of {len(admissions_df)} total")
            
            if len(filtered_df) > 0:
                # Simple table view of filtered admissions
                st.write("**Recent Patient Admissions**")
                
                recent_admissions = filtered_df.head(10).copy()
                recent_admissions['admit_time'] = recent_admissions['admit_time'].dt.strftime('%Y-%m-%d %H:%M')
                
                # Create display dataframe - FIXED: Use [] not {}
                display_df = recent_admissions[[
                    'patient_id', 'admit_time', 'ward_type', 
                    'diagnosis_code', 'med_used', 'severity_score'
                ]].copy()
                
                display_df.columns = ['Patient ID', 'Admit Time', 'Ward', 'Diagnosis', 'Medication', 'Severity']
                
                st.dataframe(display_df, width='stretch', hide_index=True)
                
                # Expandable detailed view
                st.write("**Detailed View (Click to Expand)**")
                
                for idx, row in filtered_df.iterrows():                    
                    with st.expander(f"{row['patient_id']} | {row['admit_time'].strftime('%m-%d %H:%M')} | {row['severity_score']}/25"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Patient ID:**\n{row['patient_id']}")
                            st.write(f"**Admission Time:**\n{row['admit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        with col2:
                            st.write(f"**Ward Type:**\n{row['ward_type']}")
                        
                        with col3:
                            st.write(f"**Diagnosis Code:**\n{row['diagnosis_code']}")
                            st.write(f"**Diagnosis Name:**\n{row.get('diagnosis_name', 'Unknown')}")
                        
                        st.divider()
                        
                        col4, col5, col6 = st.columns(3)
                        
                        with col4:
                            st.write(f"**Medication:**\n{row['med_used']}")
                        
                        with col5:
                            st.write(f"**Quantity:**\n{row['qty']} unit(s)")
                        
                        with col6:
                            st.write(f"**Severity Score:**\n{row['severity_score']}/25")
                        
                        st.write(f"**Estimated Stay:** {row['estimated_days']} days")
            else:
                st.info("No admissions found for the selected filter(s)")
        
        with timeline_tab2:
            # Apply same filters to analytics
            col_filter_a1, col_filter_a2 = st.columns(2)
            
            with col_filter_a1:
                filter_type_analytics = st.radio(
                    "Analytics Filter:",
                    ["All Time", "Date Range", "Specific Month"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="analytics_filter"
                )
            
            analytics_df = admissions_df.copy()
            
            with col_filter_a2:
                if filter_type_analytics == "Date Range":
                    date_range_analytics = st.date_input(
                        "Select date range for analytics",
                        value=(admissions_df['date'].min(), admissions_df['date'].max()),
                        min_value=admissions_df['date'].min(),
                        max_value=admissions_df['date'].max(),
                        key="date_range_analytics"
                    )
                    if isinstance(date_range_analytics, tuple) and len(date_range_analytics) == 2:
                        start_date_a, end_date_a = date_range_analytics
                        analytics_df = analytics_df[(analytics_df['date'] >= start_date_a) & (analytics_df['date'] <= end_date_a)]
                
                elif filter_type_analytics == "Specific Month":
                    available_months_a = sorted(admissions_df['month'].unique(), reverse=True)
                    selected_month_a = st.selectbox(
                        "Select month for analytics",
                        available_months_a,
                        format_func=lambda x: str(x),
                        key="month_filter_analytics"
                    )
                    analytics_df = analytics_df[analytics_df['month'] == selected_month_a]
            
            st.divider()
            
            st.write("**Key Metrics**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Admissions", len(analytics_df), delta="records")
            
            with col2:
                avg_severity = analytics_df['severity_score'].mean() if len(analytics_df) > 0 else 0
                severity_label = "Critical" if avg_severity >= 15 else "High" if avg_severity >= 10 else "Moderate" if avg_severity >= 5 else "Low"
                st.metric("Avg Severity", f"{avg_severity:.1f}/25", delta=severity_label)
            
            with col3:
                avg_stay = analytics_df['estimated_days'].mean() if len(analytics_df) > 0 else 0
                st.metric("Avg Stay (days)", f"{avg_stay:.1f}", delta="inpatient only")
            
            with col4:
                inpatient_count = len(analytics_df[analytics_df['estimated_days'] > 0])
                inpatient_pct = int((inpatient_count / len(analytics_df) * 100)) if len(analytics_df) > 0 else 0
                st.metric("Inpatient Rate", f"{inpatient_pct}%", delta=f"{inpatient_count} patients")
            
            st.divider()
            
            if len(analytics_df) > 0:
                # Ward distribution
                st.write("**Distribution by Ward Type**")
                ward_count = analytics_df['ward_type'].value_counts()
                
                fig_ward = px.bar(
                    x=ward_count.index,
                    y=ward_count.values,
                    title="Number of Admissions by Ward Type",
                    labels={'x': 'Ward Type', 'y': 'Number of Admissions'},
                    color=ward_count.index,
                    height=350,
                    color_discrete_map={
                        'General': '#1f77b4',
                        'ICU': '#ff7f0e',
                        'Neurological': '#2ca02c',
                        'Outpatient': '#d62728'
                    }
                )
                fig_ward.update_layout(showlegend=False)
                st.plotly_chart(fig_ward, use_container_width=True)
                
                # Top diagnoses
                st.write("**Top Diagnoses**")
                diagnosis_count = analytics_df['diagnosis_code'].value_counts().head(8)
                
                fig_diagnosis = px.bar(
                    x=diagnosis_count.values,
                    y=diagnosis_count.index,
                    orientation='h',
                    title="Most Common Diagnoses",
                    labels={'x': 'Count', 'y': 'Diagnosis Code'},
                    height=350
                )
                st.plotly_chart(fig_diagnosis, use_container_width=True)
                
                # Severity distribution
                st.write("**Severity Score Distribution**")
                severity_bins = pd.cut(analytics_df['severity_score'], bins=[0, 5, 10, 15, 25], labels=['Low (0-5)', 'Moderate (5-10)', 'High (10-15)', 'Critical (15+)'])
                severity_count = severity_bins.value_counts().sort_index()
                
                fig_severity = px.pie(
                    values=severity_count.values,
                    names=severity_count.index,
                    title="Patient Risk Distribution",
                    height=350,
                    color_discrete_sequence=['#2ca02c', '#ffd700', '#ff7f0e', '#d62728']
                )
                st.plotly_chart(fig_severity, use_container_width=True)
            else:
                st.info("No data available for the selected filter")
        
        with timeline_tab3:
            # Hospital filter
            col_hosp1, col_hosp2 = st.columns(2)
            
            with col_hosp1:
                hospital_filter = st.multiselect(
                    "Filter by Hospital",
                    admissions_df['hospital_name'].unique().tolist(),
                    default=admissions_df['hospital_name'].unique().tolist(),
                    key="hospital_filter"
                )
            
            hospital_df = admissions_df[admissions_df['hospital_name'].isin(hospital_filter)]
            
            st.write("**Hospital Statistics**")
            
            hospital_stats = hospital_df.groupby('hospital_name').agg({
                'patient_id': 'count',
                'severity_score': 'mean',
                'estimated_days': 'mean'
            }).round(2)
            
            hospital_stats.columns = ['Total Admits', 'Avg Severity', 'Avg Stay (days)']
            hospital_stats = hospital_stats.sort_values('Total Admits', ascending=False)
            
            st.dataframe(hospital_stats, width='stretch')
            
            st.divider()
            
            if len(hospital_df) > 0:
                # Hospital vs Ward cross-tabulation
                st.write("**Admissions by Hospital & Ward**")
                hospital_ward = pd.crosstab(
                    hospital_df['hospital_name'],
                    hospital_df['ward_type']
                )
                
                fig_heatmap = px.imshow(
                    hospital_ward,
                    labels=dict(x='Ward Type', y='Hospital', color='Admissions'),
                    title="Hospital-Ward Admission Heatmap",
                    height=400,
                    color_continuous_scale='YlOrRd'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.dataframe(hospital_ward, width='stretch')
    
    else:
        st.info("📭 No admission data yet. Admissions will appear here as patients are admitted.")

except ImportError:
    st.warning("Supabase client not configured. Timeline feature unavailable.")
except Exception as e:
    st.error(f"Error loading timeline: {str(e)}")