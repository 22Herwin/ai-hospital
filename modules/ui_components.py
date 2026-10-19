"""
UI Components - Sidebar, Forms, and Display Functions
"""
import os
import streamlit as st
import datetime
from dotenv import load_dotenv
from .config import (
    CHUTES_MODEL, HOSPITALIZATION_DISEASES, 
    OUTPATIENT_DISEASES, HEALTHY_CODE
)
from .data_manager import load_hospitals


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title='AI Hospital Management System',
        layout='wide'
    )


def initialize_session_state():
    """Initialize session state variables"""
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = None
    if 'admission_complete' not in st.session_state:
        st.session_state.admission_complete = False
    if 'last_admission_id' not in st.session_state:
        st.session_state.last_admission_id = None
    if 'ai_unavailable' not in st.session_state:
        st.session_state['ai_unavailable'] = False
    
    # Check if AI token is missing
    if not os.getenv("CHUTES_API_TOKEN"):
        st.session_state['ai_unavailable'] = True


def safe_rerun():
    """Safe rerun helper for different Streamlit versions"""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.session_state['_safe_rerun_toggle'] = not st.session_state.get('_safe_rerun_toggle', False)


def render_sidebar():
    """Render sidebar with system information and disease reference"""
    st.sidebar.header('System Information')
    
    # Disease reference guide
    with st.sidebar.expander("Disease Reference Guide", expanded=False):
        st.subheader("Hospitalization Required")
        for code, val in HOSPITALIZATION_DISEASES.items():
            display = val["name"] if isinstance(val, dict) else str(val)
            st.markdown("**" + str(code) + "**: " + display)

        st.subheader("Outpatient Management")
        for code, val in OUTPATIENT_DISEASES.items():
            display = val["name"] if isinstance(val, dict) else str(val)
            st.markdown("**" + str(code) + "**: " + display)

        st.subheader("Healthy")
        st.write(f"**Z00.0**: General medical examination")
    
    # AI status
    if st.session_state.get('ai_unavailable', False):
        st.sidebar.error(f"{CHUTES_MODEL} unavailable — using fallback rules")
        if st.sidebar.button("Retry AI Connection"):
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"), override=True)
            if os.getenv("CHUTES_API_TOKEN"):
                st.session_state['ai_unavailable'] = False
            safe_rerun()
    else:
        st.sidebar.success(f"Chutes AI Connected")
        st.sidebar.info(f"Model: {CHUTES_MODEL}")
    
    # Hospital occupancy status
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
    
    st.sidebar.info("""
**Clinical Decision Support**
- Specific disease categorization
- ICD-10 code validation
- Hospitalization criteria
- Evidence-based treatment
""")


def render_main_header():
    """Render main page header"""
    st.title('AI Hospital Management System')
    st.markdown("### Specific Disease Diagnosis & Patient Management")


def render_patient_form():
    """Render patient input form and return form data when submitted"""
    from .data_manager import generate_sequential_patient_id
    
    with st.form('patient_form'):
        st.subheader("Patient Clinical Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
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
        return {
            'submitted': True,
            'pid': pid,
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'blood_pressure_sys': bp_sys,
            'blood_pressure_dia': bp_dia,
            'heart_rate': hr,
            'temperature': temp,
            'symptom_cough': cough,
            'symptom_fever': fever,
            'symptom_breathless': breathless,
            'symptom_chest_pain': chest_pain,
            'symptom_neuro': neuro,
            'comorbidity_diabetes': diabetes,
            'comorbidity_hypertension': hypertension,
            'lab_wbc': wbc,
            'lab_crp': crp
        }
    return {'submitted': False}


def display_analysis_results(patient: dict, ai_result: dict):
    """Display clinical analysis results"""
    from .config import HOSPITALIZATION_DISEASES, OUTPATIENT_DISEASES
    
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
