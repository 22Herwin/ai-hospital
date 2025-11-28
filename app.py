from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
import datetime
import time
import json
import re  # ensure re is imported near top (you already had it in file; keep this)
from typing import Any, Dict, List, Optional, Tuple
import ollama  # Import Ollama library for local inference

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Ollama configuration - no API keys needed!
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "meditron:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Initialize session state
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None
if 'admission_complete' not in st.session_state:
    st.session_state.admission_complete = False
if 'last_admission_id' not in st.session_state:
    st.session_state.last_admission_id = None
# track AI availability so we don't keep calling after errors
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

# Mock diagnosis mapping for fallback
DIAGNOSIS_MAPPING = {
    'D01': {'name': 'Pneumonia', 'medicines': ['Amoxicillin 500mg', 'Azithromycin 250mg']},
    'D02': {'name': 'Hypertension', 'medicines': ['Amlodipine 5mg', 'Lisinopril 10mg']},
    'D03': {'name': 'Diabetes', 'medicines': ['Metformin 500mg', 'Insulin Glargine']},
    'D04': {'name': 'Influenza', 'medicines': ['Oseltamivir 75mg', 'Paracetamol 500mg']},
    'D05': {'name': 'Gastroenteritis', 'medicines': ['Ondansetron 4mg', 'Loperamide 2mg']},
    'D06': {'name': 'Back Pain', 'medicines': ['Ibuprofen 400mg', 'Acetaminophen 650mg']}
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
            # Create default stock if file doesn't exist
            meds = []
            for v in DIAGNOSIS_MAPPING.values():
                meds.extend(v.get('medicines', []))
            meds = sorted(list(set(meds)))
            df = pd.DataFrame({'medicine_name': meds, 'stock': [10]*len(meds)})
            df.to_csv(STOCK_CSV, index=False)
            return df
    except Exception as e:
        st.error(f"Error loading stock: {str(e)}")
        return pd.DataFrame(columns=['medicine_name', 'stock'])

def save_stock(df):
    """Save updated stock to CSV"""
    try:
        df.to_csv(STOCK_CSV, index=False)
        load_stock.clear()  # Clear cache
        return True
    except Exception as e:
        st.error(f"Error saving stock: {str(e)}")
        return False

# Ollama Meditron integration
def _extract_raw_ai_text(response: Any) -> str:
    # Try a few known shapes returned by ollama / wrappers
    try:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            # common shapes: {'message': {'content': '...'}}, {'choices':[{'message':{'content':...}}]}
            if 'message' in response and isinstance(response['message'], dict):
                c = response['message'].get('content')
                if isinstance(c, str):
                    return c
            if 'choices' in response and isinstance(response['choices'], list) and len(response['choices'])>0:
                first = response['choices'][0]
                if isinstance(first, dict):
                    # nested styles
                    if 'message' in first and isinstance(first['message'], dict):
                        c = first['message'].get('content')
                        if isinstance(c, str):
                            return c
                    if 'text' in first and isinstance(first['text'], str):
                        return first['text']
                    if 'content' in first and isinstance(first['content'], str):
                        return first['content']
            # some clients return {'content': '...'}
            if 'content' in response and isinstance(response['content'], str):
                return response['content']
            # as last resort stringify the dict
            return json.dumps(response)
    except Exception:
        pass
    try:
        return str(response)
    except Exception:
        return ""

def _normalize_result(parsed: Optional[dict], raw_text: str, features: Optional[dict] = None) -> dict:
    """
    Ensure the AI response has all expected keys and sensible defaults.
    If parsed is missing critical keys (icd10_code/diagnosis_name), caller can provide
    a `features` dict; this function will not run fallback analysis itself but will
    set better defaults (e.g., 'Z00.0'/'Healthy') for low-severity cases.
    """
    # Defensive: ensure parsed is a dict to avoid type-checker/None issues
    if not isinstance(parsed, dict):
        parsed = {}

    out = {}
    # Primary fields - use several possible names, keep 'Unknown' only as last resort
    out['icd10_code'] = parsed.get('icd10_code') or parsed.get('code') or parsed.get('diagnosis_code') or None
    out['diagnosis_name'] = parsed.get('diagnosis_name') or parsed.get('diagnosis') or None

    # inpatient
    if parsed.get('inpatient') is None:
        out['inpatient'] = False
    else:
        out['inpatient'] = bool(parsed.get('inpatient'))

    # estimated_stay_days - robust parsing of different types
    try:
        val = parsed.get('estimated_stay_days')
        if val is None or val == '':
            out['estimated_stay_days'] = 0
        else:
            if isinstance(val, (int, float)):
                out['estimated_stay_days'] = int(val)
            elif isinstance(val, str):
                try:
                    out['estimated_stay_days'] = int(float(val.strip()))
                except Exception:
                    m = re.search(r'\d+', val)
                    out['estimated_stay_days'] = int(m.group(0)) if m else 0
            else:
                out['estimated_stay_days'] = 0
    except Exception:
        out['estimated_stay_days'] = 0

    out['ward_type'] = parsed.get('ward_type') or parsed.get('ward') or 'General'

    # recommended_medicines: allow string -> list
    meds = parsed.get('recommended_medicines') or parsed.get('recommended_medications') or parsed.get('meds') or []
    if isinstance(meds, str):
        meds = [m.strip() for m in re.split(r'[,\|;]\s*', meds) if m.strip()]
    out['recommended_medicines'] = meds if isinstance(meds, (list, tuple)) else []

    # rationale
    out['rationale'] = parsed.get('rationale') or parsed.get('explanation') or (raw_text[:500] if raw_text else "No rationale provided")

    # raw_output for debugging
    out['raw_output'] = raw_text

    # If AI returned nothing for icd/diagnosis, try to set a safe "healthy" default
    # if features indicate low severity / no symptoms. Caller should pass features when available.
    if (not out['icd10_code'] or not out['diagnosis_name']) and features:
        no_symptoms = not (features.get('symptom_cough') or features.get('symptom_fever') or features.get('symptom_breathless'))
        low_lab = (features.get('lab_wbc', 0) <= 11 and features.get('lab_crp', 0) <= 10)
        low_vitals = (features.get('blood_pressure_sys', 0) <= 140 and features.get('blood_pressure_dia', 0) <= 90 and features.get('temperature', 0) <= 37.5)
        if no_symptoms and low_lab and low_vitals and features.get('severity_score', 0) < 5:
            # Mark as healthy check / normal exam
            out['icd10_code'] = out.get('icd10_code') or "Z00.0"
            out['diagnosis_name'] = out.get('diagnosis_name') or "Healthy / Routine check"
            out['inpatient'] = False
            out['estimated_stay_days'] = 0
            out['recommended_medicines'] = out['recommended_medicines'] or []
            out['rationale'] = out['rationale'] or "No clinical features suggesting acute disease."
    # Final safe types
    if out['icd10_code'] is None:
        out['icd10_code'] = "Unknown"
    if out['diagnosis_name'] is None:
        out['diagnosis_name'] = "Unknown"

    return out


def analyze_with_ollama(features: dict) -> dict:
    """
    Analyze patient features using Ollama with Meditron model.
    Improved robustness: if AI returns empty/malformed content we fall back to rule-based
    diagnosis and merge results (AI fields take precedence when present).
    """
    # If previously marked unavailable, use fallback immediately
    if st.session_state.get('ai_unavailable', False):
        # fallback_analysis in this file expects features dict
        return fallback_analysis(features)

    # Build clinical note (same as before)
    clinical_note = f"""
    Patient Clinical Summary:
    Age: {features['age']} years, BMI: {features['bmi']}
    Vital Signs:
    - Blood Pressure: {features['blood_pressure_sys']}/{features['blood_pressure_dia']} mmHg
    - Heart Rate: {features['heart_rate']} bpm
    - Temperature: {features['temperature']} °C
    Symptoms:
    - Cough: {'Yes' if features['symptom_cough'] else 'No'}
    - Fever: {'Yes' if features['symptom_fever'] else 'No'}
    - Breathlessness: {'Yes' if features['symptom_breathless'] else 'No'}
    Comorbidities:
    - Diabetes: {'Yes' if features['comorbidity_diabetes'] else 'No'}
    - Hypertension: {'Yes' if features['comorbidity_hypertension'] else 'No'}
    Lab Results:
    - WBC Count: {features['lab_wbc']} x10^9/L
    - CRP Level: {features['lab_crp']} mg/L
    Clinical Severity Score: {features['severity_score']}
    """

    prompt = f"""
    You are an expert clinical decision support system specializing in internal medicine. Analyze this patient case and provide a structured response in JSON format with these EXACT keys:
    - icd10_code: Primary ICD-10 diagnosis code (e.g., "J18.9")
    - diagnosis_name: Full diagnosis name matching ICD-10 code
    - inpatient: Boolean (true/false) for hospitalization recommendation
    - estimated_stay_days: Integer for estimated hospital days if admitted
    - ward_type: String describing ward type (e.g., "General", "ICU", "Cardiac")
    - recommended_medicines: List of 2-3 specific medication recommendations with dosages
    - rationale: Brief clinical justification for recommendations (max 100 words)

    PATIENT DATA:
    {clinical_note}

    IMPORTANT: Return ONLY valid JSON with no additional text or explanation. Do not include any text outside the JSON structure.
    """

    try:
        with st.spinner(f"Analyzing with {OLLAMA_MODEL} (local)..."):
            client = ollama.Client(host=OLLAMA_HOST)
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a clinical decision support AI that responds ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                format="json",
                options={
                    "temperature": 0.3,
                    "num_ctx": 4096
                }
            )

        ai_raw = _extract_raw_ai_text(response)

        # log raw output (same as before)
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            log_path = os.path.join(DATA_DIR, "ollama_raw_responses.log")
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.datetime.now().isoformat()} - RAW:\n{ai_raw}\n\n")
        except Exception:
            pass

        # Try to parse JSON
        parsed = None
        try:
            parsed = json.loads(ai_raw)
        except Exception:
            # extract first {...} block
            m = re.search(r'(\{[\s\S]*\})', ai_raw)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = None

        # If still not parsed, try heuristic line parsing
        if parsed is None:
            parsed = {}
            for line in ai_raw.splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    k = k.strip().lower().replace(' ', '_')
                    v = v.strip()
                    if v:
                        if ',' in v:
                            parsed[k] = [x.strip() for x in v.split(',') if x.strip()]
                        else:
                            parsed[k] = v

        # If parsed is empty or lacks key decision fields, use fallback and merge:
        missing_core = not parsed or (not parsed.get('icd10_code') and not parsed.get('diagnosis_name') and not parsed.get('diagnosis'))
        if missing_core:
            # Use fallback rule-based diagnosis for safety
            fallback = fallback_analysis(features)
            # Normalize the parsed (even if empty) with features to get types & defaults
            normalized_ai = _normalize_result(parsed if isinstance(parsed, dict) else {}, ai_raw, features)
            # Merge: prefer AI when it provided a value, else use fallback values
            merged = {}
            merged['icd10_code'] = normalized_ai.get('icd10_code') if normalized_ai.get('icd10_code') != "Unknown" else fallback.get('icd10_code')
            merged['diagnosis_name'] = normalized_ai.get('diagnosis_name') if normalized_ai.get('diagnosis_name') != "Unknown" else fallback.get('diagnosis_name')
            merged['inpatient'] = normalized_ai.get('inpatient') if normalized_ai.get('inpatient') is not None else fallback.get('inpatient')
            merged['estimated_stay_days'] = normalized_ai.get('estimated_stay_days') if normalized_ai.get('estimated_stay_days') not in (None, 0) else fallback.get('estimated_stay_days')
            merged['ward_type'] = normalized_ai.get('ward_type') if normalized_ai.get('ward_type') not in (None, '', 'General') else fallback.get('ward_type')
            merged['recommended_medicines'] = normalized_ai.get('recommended_medicines') or fallback.get('recommended_medicines')
            merged['rationale'] = normalized_ai.get('rationale') or fallback.get('rationale')
            merged['raw_output'] = ai_raw
            return merged

        # If AI responded OK, normalize and return (give AI precedence)
        normalized = _normalize_result(parsed if isinstance(parsed, dict) else {}, ai_raw, features)
        return normalized

    except Exception as e:
        error_msg = str(e)
        st.error(f"Ollama analysis failed: {error_msg}")
        # mark AI unavailable on connection/model errors
        if "connection" in error_msg.lower() or "model" in error_msg.lower():
            st.session_state['ai_unavailable'] = True
            st.sidebar.error(f"Ollama server/model unavailable — switched to fallback rules.")
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                log_path = os.path.join(DATA_DIR, "ollama_errors.log")
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{datetime.datetime.now().isoformat()} - Ollama Error: {error_msg}\n")
            except Exception:
                pass
        return fallback_analysis(features)

def fallback_analysis(features: dict) -> dict:
    """Fallback rule-based analysis when AI fails"""
    st.warning("Using fallback clinical rules (AI unavailable)")
    
    # Simple rule-based diagnosis
    if features['symptom_cough'] and features['symptom_fever'] and features['lab_wbc'] > 11:
        diagnosis = {'code': 'J18.9', 'name': 'Pneumonia', 'meds': ['Amoxicillin 500mg', 'Azithromycin 250mg']}
    elif features['comorbidity_hypertension'] and features['blood_pressure_sys'] > 160:
        diagnosis = {'code': 'I10', 'name': 'Hypertension', 'meds': ['Amlodipine 5mg', 'Lisinopril 10mg']}
    elif features['comorbidity_diabetes'] and features['lab_crp'] > 20:
        diagnosis = {'code': 'E11.9', 'name': 'Type 2 Diabetes', 'meds': ['Metformin 500mg', 'Insulin Glargine']}
    else:
        diagnosis = {'code': 'R50.9', 'name': 'Fever of unknown origin', 'meds': ['Paracetamol 500mg', 'Ibuprofen 400mg']}
    
    # Simple admission rules
    inpatient = features['severity_score'] >= 10
    stay_days = 5 if inpatient else 0
    ward = "ICU" if features['severity_score'] >= 15 else "General"
    
    return {
        "icd10_code": diagnosis['code'],
        "diagnosis_name": diagnosis['name'],
        "inpatient": inpatient,
        "estimated_stay_days": stay_days,
        "ward_type": ward,
        "recommended_medicines": diagnosis['meds'],
        "rationale": "Fallback rule-based analysis due to AI unavailability"
    }

# Calculate severity score (same logic as before)
def calc_severity_score(features: dict) -> int:
    score = 0
    # Vital signs
    if features['blood_pressure_sys'] > 180 or features['blood_pressure_sys'] < 90: score += 2
    if features['blood_pressure_dia'] > 120 or features['blood_pressure_dia'] < 60: score += 2
    if features['heart_rate'] > 120 or features['heart_rate'] < 50: score += 2
    if features['temperature'] > 39.0: score += 2
    
    # Symptoms
    if features['symptom_cough'] or features['symptom_fever'] or features['symptom_breathless']: score += 1
    if features['symptom_cough'] and features['symptom_fever']: score += 1
    if features['symptom_cough'] and features['symptom_breathless']: score += 1
    if features['symptom_fever'] and features['symptom_breathless']: score += 2
    
    # Lab values
    if features['lab_wbc'] > 15.0: score += 3
    elif features['lab_wbc'] > 11.0: score += 2
    if features['lab_crp'] > 50: score += 3
    elif features['lab_crp'] > 20: score += 2
    elif features['lab_crp'] > 10: score += 1
    
    # Comorbidities
    if features['comorbidity_diabetes']: score += 1
    if features['comorbidity_hypertension']: score += 1
    if features['age'] > 65: score += 1
    
    return score

# Sidebar configuration
st.sidebar.header('System Information')
# indicate AI availability and provide operator actions when unavailable
if st.session_state.get('ai_unavailable', False):
    st.sidebar.error(f"{OLLAMA_MODEL} unavailable — switched to local fallback rules.")
    with st.sidebar.expander("Ollama status & actions", expanded=False):
        log_path = os.path.join(DATA_DIR, "ollama_errors.log")
        if os.path.exists(log_path):
            if st.button("Show Ollama error log"):
                try:
                    with open(log_path, "r", encoding="utf-8") as fh:
                        log_text = fh.read()
                except Exception as _e:
                    log_text = f"Failed to read log: {_e}"
                st.code(log_text, language="text")
        else:
            st.write("No ollama_errors.log found.")

        if st.button("Mark AI as available (clear flag)"):
            st.session_state['ai_unavailable'] = False
            st.sidebar.success("AI marked available. Re-run analysis to attempt again.")
            safe_rerun()

        st.markdown(
            "Operator guidance: Ensure Ollama server is running and model is pulled. "
            "Run `ollama serve` in terminal and `ollama pull meditron:7b` to install the model."
        )
else:
    st.sidebar.success(f"Ollama Connected ({OLLAMA_HOST})")
    st.sidebar.info(f"Using model: {OLLAMA_MODEL}")

st.sidebar.info("""
**AI-Powered Hospital System**
- Uses Ollama with Meditron for local clinical decision support
- No API keys or internet connection required
- All patient data stays on your machine
""")

# Main content
st.title('AI Hospital Management System (Ollama Meditron)')
st.markdown("### Intelligent Patient Admission & Resource Management")

# Patient input form
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
        diabetes = st.checkbox('Diabetes')
        hypertension = st.checkbox('Hypertension')
        wbc = st.number_input('WBC Count (10^9/L)', value=7.0, step=0.1, min_value=0.0, max_value=50.0)
        crp = st.number_input('CRP Level (mg/L)', value=5.0, step=0.1, min_value=0.0, max_value=300.0)
    
    submitted = st.form_submit_button(f'Analyze with {OLLAMA_MODEL}', type='primary')

if submitted:
    # Calculate severity score
    features = {
        'age': age, 'bmi': bmi,
        'blood_pressure_sys': bp_sys, 'blood_pressure_dia': bp_dia,
        'heart_rate': hr, 'temperature': temp,
        'symptom_cough': int(cough), 'symptom_fever': int(fever), 'symptom_breathless': int(breathless),
        'comorbidity_diabetes': int(diabetes), 'comorbidity_hypertension': int(hypertension),
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
    
    # Analyze with Ollama AI (or fallback if previously flagged)
    if st.session_state.get('ai_unavailable', False):
        ai_result = fallback_analysis(features)
    else:
        ai_result = analyze_with_ollama(features)
    
    if ai_result:
        st.session_state.current_patient['ai_result'] = ai_result
        st.session_state.current_patient['diagnosis_code'] = ai_result.get('icd10_code', 'Unknown')
        st.session_state.current_patient['diagnosis_name'] = ai_result.get('diagnosis_name', 'Unknown')
    
    st.session_state.admission_complete = False
    safe_rerun()

# Display analysis results
if st.session_state.current_patient and not st.session_state.admission_complete:
    patient = st.session_state.current_patient
    features = patient['features']
    ai_result = patient.get('ai_result', {})
    
    st.subheader(f"{OLLAMA_MODEL} Clinical Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Patient Features Analyzed:**")
        st.json(features)
    
    with col2:
        st.metric("Clinical Severity Score", patient['severity_score'], 
                 delta="Critical" if patient['severity_score'] >= 15 else 
                       "High Risk" if patient['severity_score'] >= 10 else 
                       "Moderate Risk" if patient['severity_score'] >= 5 else "Low Risk",
                 delta_color="inverse")
        
        # Severity interpretation
        score = patient['severity_score']
        if score >= 15:
            st.error("Critical Condition - Requires immediate intervention")
        elif score >= 10:
            st.warning("High Risk - Close monitoring needed")
        elif score >= 5:
            st.info("Moderate Risk - Regular monitoring recommended")
        else:
            st.success("Low Risk - Outpatient management appropriate")
    
    # AI Results Display
    st.markdown("---")
    st.subheader("AI Clinical Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if ai_result:
            st.success(f"**Diagnosis:** {ai_result.get('diagnosis_name', 'Unknown')} ({ai_result.get('icd10_code', 'Unknown')})")
            st.info(f"**Hospitalization Recommended:** {'Yes' if ai_result.get('inpatient') else 'No'}")
            if ai_result.get('inpatient'):
                st.warning(f"**Recommended Ward:** {ai_result.get('ward_type', 'General')}")
                st.info(f"**Estimated Stay:** {ai_result.get('estimated_stay_days', 3)} days")
        else:
            st.error("AI analysis failed - using fallback rules")
    
    with col2:
        st.subheader("Recommended Medications")
        if ai_result and ai_result.get('recommended_medicines'):
            for med in ai_result['recommended_medicines']:
                st.write(f"- {med}")
        else:
            st.write("No recommendations available")
    
    # Rationale section
    with st.expander("Clinical Rationale (AI Explanation)"):
        if ai_result:
            st.write(ai_result.get('rationale', 'No rationale provided'))
        else:
            st.write("Fallback rule-based analysis used")
    
    # Admission workflow
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

# Inventory management section
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
    if st.button('Replenish Low Stock (+10 units)'):
        low_stock = stock_df[stock_df['stock'] < 5]
        if not low_stock.empty:
            for idx in low_stock.index:
                # Ensure the stored value is numeric before adding
                current_val = pd.to_numeric(stock_df.at[idx, 'stock'], errors='coerce')
                if pd.isna(current_val):
                    current_val = 0
                stock_df.at[idx, 'stock'] = int(current_val) + 10
            save_stock(stock_df)
            st.success(f"Replenished {len(low_stock)} low-stock items")
        else:
            st.info("All items have sufficient stock")

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
st.caption(f"AI Hospital Management System • Powered by {OLLAMA_MODEL} • Local Inference • © 2025")