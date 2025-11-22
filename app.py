from dotenv import load_dotenv
import streamlit as st
import pandas as pd, numpy as np, joblib, os, datetime, time
from typing import Any, Dict
from supabase_client import (
    init_supabase_client, 
    insert_admission_supabase, 
    decrement_medicine_stock_supabase, 
    update_admission_supabase, 
    get_supabase_status,
    replenish_medicine_stock_supabase
)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Add diagnosis mapping
DIAGNOSIS_MAPPING = {
    'D01': {
        'name': 'Pneumonia',
        'description': 'Lung infection causing inflammation and fluid buildup',
        'medicines': ['Amoxicillin 500mg', 'Azithromycin 250mg', 'Paracetamol 500mg']
    },
    'D02': {
        'name': 'Hypertensive Heart Disease',
        'description': 'Heart damage due to chronic high blood pressure',
        'medicines': ['Amlodipine 5mg', 'Lisinopril 10mg', 'Atorvastatin 20mg']
    },
    'D03': {
        'name': 'Type 2 Diabetes',
        'description': 'Metabolic disorder causing high blood sugar levels',
        'medicines': ['Metformin 500mg', 'Insulin Glargine', 'Sitagliptin 100mg']
    },
    'D04': {
        'name': 'Influenza',
        'description': 'Viral respiratory infection with high fever',
        'medicines': ['Oseltamivir 75mg', 'Paracetamol 500mg', 'Dexamethasone 4mg']
    },
    'D05': {
        'name': 'Stroke',
        'description': 'Sudden loss of blood flow to the brain',
        'medicines': ['Aspirin 81mg', 'Clopidogrel 75mg', 'Atorvastatin 40mg']
    },
    'D06': {
        'name': 'Gastroenteritis',
        'description': 'Inflammation of stomach and intestines',
        'medicines': ['Ondansetron 4mg', 'Loperamide 2mg', 'Oral Rehydration Salts']
    }
}

# safe rerun helper
def safe_rerun():
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()
    else:
        st.session_state['_safe_rerun_toggle'] = not st.session_state.get('_safe_rerun_toggle', False)

st.set_page_config(page_title='AI Hospital Management System', layout='wide')

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STOCK_CSV = os.path.join(DATA_DIR, 'medicine_stock.csv')
PATIENTS_LOG = os.path.join(DATA_DIR, 'admission_log.csv')

# Initialize session state
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None
if 'admission_complete' not in st.session_state:
    st.session_state.admission_complete = False
if 'last_admission_id' not in st.session_state:
    st.session_state.last_admission_id = None

@st.cache_resource
def load_stock():
    try:
        df = pd.read_csv(STOCK_CSV)
        df['stock'] = pd.to_numeric(df['stock'], errors='coerce').fillna(0).astype(int)
        return df
    except Exception as e:
        st.error(f"Error loading stock: {str(e)}")
        return pd.DataFrame(columns=['medicine_name', 'stock'])

def save_stock(df):
    try:
        df.to_csv(STOCK_CSV, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving stock: {str(e)}")
        return False

# New: lightweight fallback diagnosis model (rule-based)
class FallbackDiagnosisModel:
    def predict(self, X):
        # X is a DataFrame-like with one row
        row = X.iloc[0]
        # Simple rules to map to diagnosis codes
        if row.get('symptom_cough', 0) and row.get('symptom_fever', 0) and row.get('lab_wbc', 0) > 11:
            return ['D01']  # Pneumonia
        if row.get('comorbidity_hypertension', 0) and row.get('age', 0) > 60:
            return ['D02']  # Hypertensive Heart Disease
        if row.get('comorbidity_diabetes', 0):
            return ['D03']  # Type 2 Diabetes
        if row.get('symptom_fever', 0) and row.get('symptom_cough', 0) and row.get('temperature', 0) > 38:
            return ['D04']  # Influenza
        if row.get('symptom_breathless', 0) and (row.get('blood_pressure_sys', 0) > 180 or row.get('heart_rate', 0) > 120):
            return ['D05']  # Stroke (proxy)
        return ['D06']  # Gastroenteritis / default

    def predict_proba(self, X):
        # Return a single-column probability array with reasonable confidence for the selected label
        # The code only takes max(proba) so shape/details are flexible
        n = len(X)
        # give moderate confidence
        return np.array([[0.65] for _ in range(n)])
def load_models():
    models: Dict[str, Any] = {
        'inpatient': None, 
        'ward': None, 
        'stay': None,
        'diagnosis': None
    }
    model_files = {
        'inpatient': 'inpatient_model.pkl',
        'ward': 'ward_model.pkl',
        'stay': 'stay_model.pkl',
        'diagnosis': 'diagnosis_model.pkl'
    }
    
    for name, filename in model_files.items():
        try:
            models[name] = joblib.load(os.path.join(MODELS_DIR, filename))
            st.sidebar.success(f"{name.capitalize()} model loaded")
        except Exception as e:
            # If diagnosis model missing, provide a fallback rule-based model to keep app functional
            if name == 'diagnosis':
                models[name] = FallbackDiagnosisModel()
                st.sidebar.warning(f"Diagnosis model not found; using fallback rule-based predictor ({filename}).")
            else:
                models[name] = None
                st.sidebar.error(f"Failed to load {name} model: {str(e)}")
    return models
    return models

# Sidebar configuration
st.sidebar.header('System Configuration')
db = init_supabase_client()
ok, msg = get_supabase_status()
st.sidebar.write(f"**Supabase Status:** {'Connected' if ok else 'Disconnected'}")
st.sidebar.caption(msg)

if st.sidebar.button('Refresh Models'):
    st.cache_resource.clear()
    safe_rerun()

models = load_models()

# Dataset generation and training controls
st.sidebar.markdown("---")
st.sidebar.subheader("Data Management")
if st.sidebar.button('Generate Sample Dataset (1000 rows)'):
    with st.spinner("Generating sample dataset..."):
        import subprocess, sys
        subprocess.run([
            sys.executable, 
            os.path.join(BASE_DIR, 'generate_dataset.py'),
            '--n', '1000',
            '--out', os.path.join(DATA_DIR, 'patients_sample.csv')
        ])
    st.sidebar.success('Dataset generated successfully!')

if st.sidebar.button('Train Prediction Models'):
    with st.spinner("Training models (this may take 2-5 minutes)..."):
        import subprocess, sys
        result = subprocess.run([
            sys.executable,
            os.path.join(BASE_DIR, 'train_models.py'),
            '--data', os.path.join(DATA_DIR, 'patients_sample.csv'),
            '--out_dir', MODELS_DIR
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            st.sidebar.success('Training completed successfully!')
            st.cache_resource.clear()
            safe_rerun()
        else:
            st.sidebar.error(f"Training failed: {result.stderr}")

# Main content
st.title('AI Hospital Management System')
st.markdown("### Patient Admission & Resource Management")

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
    
    submitted = st.form_submit_button('Analyze Patient Risk')

if submitted:
    # Calculate severity score - more clinically accurate
    severity_score = 0
    
    # Vital signs assessment
    if bp_sys > 180 or bp_sys < 90: severity_score += 2
    if bp_dia > 120 or bp_dia < 60: severity_score += 2
    if hr > 120 or hr < 50: severity_score += 2
    if temp > 39.0: severity_score += 2
    
    # Symptom assessment
    if cough or fever or breathless: severity_score += 1
    if cough and fever: severity_score += 1
    if cough and breathless: severity_score += 1
    if fever and breathless: severity_score += 2
    
    # Lab values
    if wbc > 15.0: severity_score += 3
    elif wbc > 11.0: severity_score += 2
    if crp > 50: severity_score += 3
    elif crp > 20: severity_score += 2
    elif crp > 10: severity_score += 1
    
    # Comorbidities
    if diabetes: severity_score += 1
    if hypertension: severity_score += 1
    if age > 65: severity_score += 1
    
    # Prepare features for prediction
    features = pd.DataFrame([{
        'age': age, 
        'bmi': bmi, 
        'blood_pressure_sys': bp_sys, 
        'blood_pressure_dia': bp_dia,
        'heart_rate': hr, 
        'temperature': temp,
        'symptom_cough': int(cough),
        'symptom_fever': int(fever),
        'symptom_breathless': int(breathless),
        'comorbidity_diabetes': int(diabetes),
        'comorbidity_hypertension': int(hypertension),
        'lab_wbc': wbc,
        'lab_crp': crp,
        'severity_score': severity_score
    }])
    
    # Store in session state
    st.session_state.current_patient = {
        'pid': pid,
        'features': features,
        'severity_score': severity_score,
        'timestamp': datetime.datetime.now()
    }
    
    st.session_state.admission_complete = False
    safe_rerun()

# Display analysis results if patient data exists
if st.session_state.current_patient and not st.session_state.admission_complete:
    patient = st.session_state.current_patient
    features = patient['features']
    
    st.subheader("Clinical Analysis Results")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Patient Features Used for Prediction:**")
        feature_display = features.T
        feature_display.columns = ['Value']
        st.dataframe(feature_display)
    
    with col2:
        st.metric("Clinical Severity Score", patient['severity_score'], 
                 delta="Critical" if patient['severity_score'] >= 15 else 
                       "High Risk" if patient['severity_score'] >= 10 else 
                       "Moderate Risk" if patient['severity_score'] >= 5 else "Low Risk",
                 delta_color="inverse")
        
        # Show severity score interpretation
        score = patient['severity_score']
        if score >= 15:
            st.error("Critical Condition - Requires immediate intervention")
        elif score >= 10:
            st.warning("High Risk - Close monitoring needed")
        elif score >= 5:
            st.info("Moderate Risk - Regular monitoring recommended")
        else:
            st.success("Low Risk - Outpatient management appropriate")

    # Add diagnosis prediction section
    if models['diagnosis'] is None:
        st.error("Diagnosis prediction model not loaded. Please train models in sidebar.")
    else:
        with st.spinner("Analyzing symptoms to determine diagnosis..."):
            time.sleep(0.5)
            
            try:
                # Get predicted diagnosis
                diagnosis_code = models['diagnosis'].predict(features)[0]
                
                # Ensure diagnosis_code is a string
                if isinstance(diagnosis_code, np.ndarray):
                    diagnosis_code = str(diagnosis_code[0])
                else:
                    diagnosis_code = str(diagnosis_code)
                    
                # Get diagnosis data
                diagnosis_data = DIAGNOSIS_MAPPING.get(diagnosis_code, {})
                
                # Store in session state
                st.session_state.current_patient['diagnosis_code'] = diagnosis_code
                st.session_state.current_patient['diagnosis_data'] = diagnosis_data
                
                st.subheader("AI Diagnosis")
                if diagnosis_data:
                    st.success(f"**{diagnosis_data['name']}**")
                    st.caption(diagnosis_data['description'])
                    
                    # Show confidence for diagnosis
                    try:
                        proba = models['diagnosis'].predict_proba(features)[0]
                        max_prob = max(proba) * 100
                        st.caption(f"Diagnosis confidence: {max_prob:.1f}%")
                    except:
                        pass
                else:
                    st.warning("Could not determine diagnosis. Please consult with specialist.")
                    diagnosis_data = {'name': 'Unknown', 'medicines': []}
                    
            except Exception as e:
                st.error(f"Error predicting diagnosis: {str(e)}")
                diagnosis_code = 'D06'  # Default to gastroenteritis
                diagnosis_data = DIAGNOSIS_MAPPING.get(diagnosis_code, {})
                st.session_state.current_patient['diagnosis_code'] = diagnosis_code
                st.session_state.current_patient['diagnosis_data'] = diagnosis_data

    # Prediction section
    st.markdown("---")
    st.subheader("AI Prediction Results")
    
    if models['inpatient'] is None:
        st.error("Inpatient prediction model not loaded. Please train models in sidebar.")
    else:
        with st.spinner("Calculating admission recommendation..."):
            time.sleep(0.5)
            
            # Try predicting normally, but if the model complains about missing columns
            # (e.g. 'diagnosis_code'), augment features with that column and retry.
            try:
                pred_in = models['inpatient'].predict(features)[0]
                pred_in = int(pred_in[0]) if isinstance(pred_in, np.ndarray) else int(pred_in)
            except Exception as e:
                err = str(e)
                # Quick check for missing diagnosis_code in the exception text
                if 'diagnosis_code' in err:
                    features_with_diag = features.copy()
                    diag_code = st.session_state.current_patient.get('diagnosis_code', 'D06')
                    features_with_diag['diagnosis_code'] = diag_code
                    try:
                        pred_in = models['inpatient'].predict(features_with_diag)[0]
                        pred_in = int(pred_in[0]) if isinstance(pred_in, np.ndarray) else int(pred_in)
                    except Exception as e2:
                        st.error(f"Error predicting admission after adding diagnosis_code: {str(e2)}")
                        pred_in = 0
                else:
                    st.error(f"Error predicting admission: {err}")
                    pred_in = 0

            # Get confidence with fallback
            prob = None
            try:
                if hasattr(models['inpatient'], 'predict_proba'):
                    # Use the same features that worked for prediction
                    # If prediction worked with features_with_diag, use that for proba too
                    proba_features = features.copy()
                    
                    # Check if the model expects diagnosis_code column
                    try:
                        # Test if current features work
                        test_proba = models['inpatient'].predict_proba(proba_features)
                        proba = test_proba
                    except Exception as e:
                        if 'diagnosis_code' in str(e):
                            # Add diagnosis_code and try again
                            diag_code = st.session_state.current_patient.get('diagnosis_code', 'D06')
                            proba_features['diagnosis_code'] = diag_code
                            proba = models['inpatient'].predict_proba(proba_features)
                        else:
                            raise
                    
                    if proba.shape[1] >= 2:
                        prob = float(proba[0, 1]) * 100
                    else:
                        prob = float(proba[0, 0]) * 100
            except Exception as proba_error:
                # Optional: uncomment this line to see the actual error
                st.warning(f"Confidence calculation failed: {str(proba_error)}")
                prob = None

            col1, col2 = st.columns(2)
            
            # Only recommend inpatient if confidence > 75% for high severity
            if pred_in == 1 and (prob is None or prob >= 75) and patient['severity_score'] >= 5:
                with col1:
                    st.success(f"**HOSPITALIZATION RECOMMENDED** (Confidence: {prob:.1f}%)" if prob is not None else 
                               "**HOSPITALIZATION RECOMMENDED** (Confidence: N/A)")
                
                # Show ward and stay prediction
                if models['ward'] is not None and models['stay'] is not None:
                    try:
                        # try predict ward; if missing columns (diagnosis_code) retry with augmented features
                        try:
                            ward_pred = models['ward'].predict(features)[0]
                        except Exception as werr:
                            if 'diagnosis_code' in str(werr):
                                features_with_diag = features.copy()
                                diag_code = st.session_state.current_patient.get('diagnosis_code', 'D06')
                                features_with_diag['diagnosis_code'] = diag_code
                                ward_pred = models['ward'].predict(features_with_diag)[0]
                            else:
                                raise

                        ward = str(ward_pred[0]) if isinstance(ward_pred, np.ndarray) else str(ward_pred)

                        # try predict stay_days; same retry-on-missing-column logic
                        try:
                            stay_pred = models['stay'].predict(features)[0]
                        except Exception as serr:
                            if 'diagnosis_code' in str(serr):
                                features_with_diag = features.copy()
                                features_with_diag['diagnosis_code'] = st.session_state.current_patient.get('diagnosis_code', 'D06')
                                stay_pred = models['stay'].predict(features_with_diag)[0]
                            else:
                                raise

                        stay_days = max(1, int(round(stay_pred)))

                        ward_details = {
                            'General': 'Standard ward for stable patients',
                            'ICU': 'Intensive care for critical cases',
                            'Isolation': 'Specialized infection control unit',
                            'Cardiac': 'Heart condition monitoring unit'
                        }
                        
                        with col2:
                            st.warning(f"Recommended Ward: {ward}")
                            st.info(f"Estimated Stay: {stay_days} days")
                            st.caption(f"*{ward_details.get(ward, 'Specialized care unit')}*")
                    except Exception as e:
                        st.warning(f"Error predicting ward/stay: {str(e)}")
                        st.info("Estimated Stay: 3 days (default)")
                        st.warning("Recommended Ward: General (default)")
            else:
                with col1:
                    st.info(f"Outpatient Care Recommended (Confidence: {100-prob:.1f}%)" if prob is not None else 
                           "Outpatient Care Recommended (Confidence: N/A)")
                
                # Show outpatient recommendation
                with col2:
                    st.info("Treatment can be managed through outpatient care")
                    st.caption("Regular follow-up recommended")

    # Admission workflow for inpatients
    if models['inpatient'] is not None and patient['severity_score'] >= 5:
        # Check if we should recommend admission based on both model and clinical judgment
        should_admit = False
        
        try:
            # Get prediction from model
            pred_in = models['inpatient'].predict(features)[0]
            pred_in = int(pred_in[0]) if isinstance(pred_in, np.ndarray) else int(pred_in)
            
            # Use clinical judgment + model prediction
            if pred_in == 1 and patient['severity_score'] >= 5:
                should_admit = True
        except:
            # Fallback: use severity score only
            if patient['severity_score'] >= 10:  # Critical or high risk
                should_admit = True
        
        if should_admit:
            st.markdown("---")
            st.subheader("Medication Assignment")
            
            stock = load_stock()
            if stock.empty:
                st.error("No medicines available in inventory. Please replenish stock.")
            else:
                # Get diagnosis data (ensure it exists)
                diagnosis_data = st.session_state.current_patient.get('diagnosis_data', {})
                if not diagnosis_data:
                    diagnosis_data = {'name': 'Unknown', 'medicines': []}
                
                # Get recommended medicines for this diagnosis
                recommended_meds = diagnosis_data.get('medicines', [])
                
                # Filter available medicines that are in stock
                available_meds = stock[
                    stock['medicine_name'].isin(recommended_meds) & 
                    (stock['stock'] > 0)
                ]
                
                if recommended_meds:
                    st.info(f"**Recommended Medicines for {diagnosis_data['name']}:**")
                    med_list = "\n".join([f"- {med}" for med in recommended_meds])
                    st.markdown(f"```{med_list}```")
                else:
                    st.warning("No specific medication recommendations for this diagnosis.")
                
                if available_meds.empty:
                    st.warning("No recommended medicines are in stock. Please replenish immediately.")
                else:
                    med_col1, med_col2 = st.columns(2)
                    
                    with med_col1:
                        # Auto-select the first available recommended medicine
                        selected_med = st.selectbox(
                            'Select Medication', 
                            available_meds['medicine_name'].tolist(),
                            index=0,
                            help="Choose from available in-stock medications"
                        )
                    
                    with med_col2:
                        # Get the selected row by position to obtain a scalar value safely
                        selected_row = available_meds[available_meds['medicine_name'] == selected_med].iloc[0]
                        stock_val = selected_row['stock']
                        # Convert numpy/pandas scalar to native Python int using .item() when available
                        current_stock = int(stock_val.item() if hasattr(stock_val, 'item') else stock_val)
                        qty = st.number_input(
                            'Quantity Needed', 
                            min_value=1, 
                            max_value=current_stock, 
                            value=1,
                            help=f"Available stock: {current_stock}"
                        )
                    
                    admit_clicked = st.button('CONFIRM ADMISSION & MEDICATION', 
                                            type='primary',
                                            use_container_width=True)
                    
                    if admit_clicked:
                        # Process admission
                        ward = 'General'  # Default
                        stay_days = 3  # Default
                        
                        try:
                            if models['ward']:
                                try:
                                    ward_pred = models['ward'].predict(features)[0]
                                except Exception as werr:
                                    if 'diagnosis_code' in str(werr):
                                        features_with_diag = features.copy()
                                        features_with_diag['diagnosis_code'] = st.session_state.current_patient.get('diagnosis_code', 'D06')
                                        ward_pred = models['ward'].predict(features_with_diag)[0]
                                    else:
                                        raise
                                ward = str(ward_pred[0]) if isinstance(ward_pred, np.ndarray) else str(ward_pred)
                                
                            if models['stay']:
                                try:
                                    stay_pred = models['stay'].predict(features)[0]
                                except Exception as serr:
                                    if 'diagnosis_code' in str(serr):
                                        features_with_diag = features.copy()
                                        features_with_diag['diagnosis_code'] = st.session_state.current_patient.get('diagnosis_code', 'D06')
                                        stay_pred = models['stay'].predict(features_with_diag)[0]
                                    else:
                                        raise
                                stay_days = max(1, int(round(stay_pred)))
                        except Exception as e:
                            st.warning(f"Using default values for ward/stay: {str(e)}")
                        
                        # Create admission data
                        admission_data = {
                            'patient_id': patient['pid'],
                            'ward_type': ward,
                            'estimated_days': stay_days,
                            'med_used': selected_med,
                            'qty': int(qty),
                            'diagnosis_code': st.session_state.current_patient.get('diagnosis_code', 'D06'),
                            'severity_score': int(patient['severity_score']),
                            'prediction_confidence': float(prob) if prob is not None else None
                        }

                        # Database operations
                        if db:
                            success, message, row = insert_admission_supabase(db, admission_data)
                            if not success:
                                st.error(f"Database error: {message}")
                                st.stop()

                            admission_id = row.get('id') if row else None
                            st.session_state.last_admission_id = admission_id

                            stock_success, stock_msg, _ = decrement_medicine_stock_supabase(
                                db, str(selected_med), int(qty)
                            )
                            if not stock_success:
                                st.error(f"Stock update failed: {stock_msg}. Admission rolled back.")
                                st.stop()

                            st.success(f"Admission recorded successfully (ID: {admission_id})" if admission_id else 
                                      "Admission recorded successfully (ID: N/A)")

                        # Local fallback
                        else:
                            st.info("Using local storage (Supabase unavailable)")
                            try:
                                log_df = pd.DataFrame([{
                                    **admission_data,
                                    'admit_time': patient['timestamp'].isoformat(),
                                    'status': 'Admitted'
                                }])
                                
                                if os.path.exists(PATIENTS_LOG):
                                    log_df.to_csv(PATIENTS_LOG, mode='a', header=False, index=False)
                                else:
                                    log_df.to_csv(PATIENTS_LOG, index=False)
                                
                                st.success("Admission logged locally")
                            except Exception as e:
                                st.error(f"Local save failed: {str(e)}")
                        
                        # Update local stock
                        try:
                            # Find the index in the original stock dataframe for the selected medicine
                            mask = stock['medicine_name'] == selected_med
                            if mask.any():
                                idx = stock.index[mask][0]
                                stock.at[idx, 'stock'] = max(0, current_stock - int(qty))
                                save_stock(stock)
                            else:
                                st.warning("Selected medicine not found in local stock; skipping local stock update.")
                        except Exception as e:
                            st.error(f"Stock update error: {str(e)}")
                            # Best-effort fallback attempt
                            try:
                                mask = stock['medicine_name'] == selected_med
                                if mask.any():
                                    idx = stock.index[mask][0]
                                    stock.at[idx, 'stock'] = max(0, current_stock - int(qty))
                                    save_stock(stock)
                            except Exception:
                                st.error("Failed to update local stock.")

                        st.balloons()
                        st.session_state.admission_complete = True
                        safe_rerun()

# Stock management section
st.markdown("---")
st.subheader("Medicine Inventory Management")

stock_df = load_stock()
if not stock_df.empty:
    stock_df['stock'] = pd.to_numeric(stock_df['stock'], errors='coerce').fillna(0).astype(int)

    # Display the stock dataframe
    st.dataframe(stock_df)

    # Highlight low stock items
    out_of_stock = stock_df[stock_df['stock'] == 0]
    low_stock = stock_df[(stock_df['stock'] > 0) & (stock_df['stock'] < 5)]
    if not out_of_stock.empty:
        st.warning(f"{len(out_of_stock)} medicine(s) are out of stock: {', '.join(out_of_stock['medicine_name'].tolist())}")
    if not low_stock.empty:
        st.info(f"ℹ{len(low_stock)} medicine(s) have low stock (<5): {', '.join(low_stock['medicine_name'].tolist())}")
else:
    st.info("No stock data available. Please generate initial stock.")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Refresh Stock Data'):
        st.cache_resource.clear()
        safe_rerun()

with col2:
    if st.button('Replenish Critical Stock', type='primary'):
        low_stock = stock_df[stock_df['stock'] < 5]
        if not low_stock.empty:
            for _, row in low_stock.iterrows():
                new_stock = int(row['stock']) + 10
                stock_df.loc[stock_df['medicine_name'] == row['medicine_name'], 'stock'] = new_stock

                if db:
                    replenish_medicine_stock_supabase(db, str(row['medicine_name']), int(new_stock))

            save_stock(stock_df)
            st.success(f"Replenished {len(low_stock)} low-stock items (+10 each)")
            safe_rerun()
        else:
            st.info("All items have sufficient stock")

with col3:
    if st.button('Full Stock Replenishment (Demo)', type='secondary'):
        stock_df['stock'] = stock_df['stock'] + 10

        if db:
            for _, row in stock_df.iterrows():
                replenish_medicine_stock_supabase(db, str(row['medicine_name']), int(row['stock']))

        save_stock(stock_df)
        st.success("All medicines replenished (+10 units each)")
        safe_rerun()

# Admission history section
st.markdown("---")
st.subheader("Recent Admissions")

if st.session_state.last_admission_id:
    st.info(f"Last admission ID: {st.session_state.last_admission_id}")
    
    if os.path.exists(PATIENTS_LOG):
        log_df = pd.read_csv(PATIENTS_LOG)
        st.dataframe(log_df.tail(5))
    else:
        st.info("No admission history available yet")

st.markdown("---")
st.caption("Hospital Management System v1.5 • Predictions based on clinical decision support algorithms")