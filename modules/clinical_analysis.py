"""
Clinical Analysis - Diagnosis, Scoring, and AI Integration
"""
import os
import time
import datetime
import streamlit as st
import concurrent.futures
from typing import Dict
from icd10_loader import lookup_icd10
from .config import (
    CHUTES_MODEL, CHUTES_TIMEOUT, DATA_DIR,
    HOSPITALIZATION_DISEASES, OUTPATIENT_DISEASES
)


def build_clinical_note(features: dict) -> str:
    """Build comprehensive clinical note for AI analysis"""
    severity_score = features['severity_score']
    
    # Determine severity level
    if severity_score >= 20:
        severity_level = "CRITICAL - IMMEDIATE INTERVENTION REQUIRED"
    elif severity_score >= 15:
        severity_level = "SEVERE - URGENT CARE NEEDED"
    elif severity_score >= 10:
        severity_level = "MODERATE - CLOSE MONITORING"
    else:
        severity_level = "MILD - ROUTINE MANAGEMENT"
    
    # Build active symptoms list
    active_symptoms = []
    if features['symptom_fever']:
        active_symptoms.append(f"Fever ({features['temperature']}°C)")
    if features['symptom_cough']:
        active_symptoms.append("Cough")
    if features['symptom_breathless']:
        active_symptoms.append("Breathlessness")
    if features.get('symptom_chest_pain', False):
        active_symptoms.append("Chest Pain")
    if features.get('symptom_neuro', False):
        active_symptoms.append("Neurological Symptoms")
    
    symptoms_text = ", ".join(active_symptoms) if active_symptoms else "None reported"
    
    return f"""
🚨 CLINICAL SEVERITY: {severity_level} (Score: {severity_score}/25) 🚨

PATIENT CLINICAL PRESENTATION:

Demographics:
- Age: {features['age']} years
- Sex: {'Unknown' if features.get('sex') not in ['M', 'F'] else features['sex']}
- BMI: {features['bmi']:.1f}

⚠️ ACUTE PRESENTING SYMPTOMS: {symptoms_text}

Vital Signs:
- Blood Pressure: {features['blood_pressure_sys']}/{features['blood_pressure_dia']} mmHg {'(HYPERTENSIVE CRISIS)' if features['blood_pressure_sys'] > 180 else '(ELEVATED)' if features['blood_pressure_sys'] > 140 else ''}
- Heart Rate: {features['heart_rate']} bpm {'(TACHYCARDIA)' if features['heart_rate'] > 100 else ''}
- Temperature: {features['temperature']} °C {'(HIGH FEVER)' if features['temperature'] > 39 else '(FEVER)' if features['temperature'] > 38 else ''}

Laboratory Findings:
- White Blood Cell Count: {features['lab_wbc']} x10^9/L {'(VERY HIGH - SEVERE INFECTION)' if features['lab_wbc'] > 20 else '(ELEVATED - INFECTION)' if features['lab_wbc'] > 11 else '(Normal: 4-11)'}
- C-Reactive Protein: {features['lab_crp']} mg/L {'(SEVERE INFLAMMATION)' if features['lab_crp'] > 100 else '(HIGH INFLAMMATION)' if features['lab_crp'] > 50 else '(ELEVATED)' if features['lab_crp'] > 10 else '(Normal: <5)'}

Past Medical History (Chronic Conditions):
- Diabetes: {'Yes' if features['comorbidity_diabetes'] else 'No'}
- Hypertension: {'Yes' if features['comorbidity_hypertension'] else 'No'}

IMPORTANT: Focus on the ACUTE PRESENTATION and severity. If severity is CRITICAL/SEVERE, diagnose the acute condition causing current symptoms, NOT underlying chronic diseases.
"""


def calc_severity_score(features: dict) -> int:
    """Calculate clinical severity score"""
    score = 0

    # Vital signs abnormalities
    if features['blood_pressure_sys'] > 180 or features['blood_pressure_sys'] < 90:
        score += 4
    elif features['blood_pressure_sys'] > 160 or features['blood_pressure_sys'] < 100:
        score += 2

    if features['blood_pressure_dia'] > 120 or features['blood_pressure_dia'] < 60:
        score += 3
    if features['heart_rate'] > 120 or features['heart_rate'] < 50:
        score += 3
    if features['temperature'] > 39.0:
        score += 3
    elif features['temperature'] > 38.0:
        score += 2

    # Symptoms
    if features['symptom_breathless']:
        score += 4
    if features.get('symptom_chest_pain', False):
        score += 4
    if features.get('symptom_neuro', False):
        score += 5
    if features['symptom_fever']:
        score += 2
    if features['symptom_cough']:
        score += 1

    # Lab values
    if features['lab_wbc'] > 15.0:
        score += 3
    elif features['lab_wbc'] > 11.0:
        score += 2
    if features['lab_crp'] > 100:
        score += 4
    elif features['lab_crp'] > 50:
        score += 3
    elif features['lab_crp'] > 20:
        score += 2
    elif features['lab_crp'] > 10:
        score += 1

    # Comorbidities and age
    if features['comorbidity_diabetes']:
        score += 1
    if features['comorbidity_hypertension']:
        score += 1
    if features['age'] > 65:
        score += 2
    elif features['age'] > 50:
        score += 1

    return min(score, 25)


def fallback_analysis(features: dict) -> dict:
    """Fallback rule-based analysis when AI fails"""
    st.warning("Using fallback clinical rules (AI unavailable)")

    # Priority-based rule evaluation (most severe first)
    # 1. Critical conditions with multiple severe symptoms + high inflammatory markers
    if features['symptom_fever'] and features['lab_wbc'] > 20 and features['lab_crp'] > 50:
        diagnosis = {
            'code': 'A41.9',
            'name': 'Sepsis, unspecified organism',
            'meds': ['Meropenem 1g IV q8h', 'Vancomycin 1g IV q12h', 'IV Fluids', 'Norepinephrine if hypotensive'],
            'ward': 'ICU',
            'stay': 14
        }
    # 1b. Severe respiratory infection with systemic response
    elif features['symptom_fever'] and features['symptom_cough'] and features['symptom_breathless'] and (features['lab_wbc'] > 15 or features['lab_crp'] > 50):
        diagnosis = {
            'code': 'J18.9',
            'name': 'Severe pneumonia, unspecified organism',
            'meds': ['Ceftriaxone 2g IV q24h', 'Azithromycin 500mg IV q24h', 'Oxygen therapy', 'IV Fluids'],
            'ward': 'ICU',
            'stay': 10
        }
    # 1c. Sepsis with elevated CRP only
    elif features['symptom_fever'] and features['lab_crp'] > 100:
        diagnosis = {
            'code': 'A41.9',
            'name': 'Sepsis, unspecified',
            'meds': ['Meropenem 1g IV', 'IV Fluids'],
            'ward': 'ICU',
            'stay': 14
        }
    # 2. Neurological emergencies (highest priority for neuro symptoms)
    elif features.get('symptom_neuro', False) and (features['heart_rate'] > 100 or features['blood_pressure_sys'] > 160):
        diagnosis = {
            'code': 'I63.9',
            'name': 'Cerebral infarction (Stroke)',
            'meds': ['Aspirin 100mg', 'Atorvastatin 40mg', 'Oxygen therapy'],
            'ward': 'Neurological',
            'stay': 10
        }
    elif features.get('symptom_neuro', False):
        diagnosis = {
            'code': 'I63.9',
            'name': 'Cerebral infarction',
            'meds': ['Aspirin 100mg', 'Atorvastatin 40mg'],
            'ward': 'Neurological',
            'stay': 10
        }
    # 3. Cardiac emergencies with chest pain
    elif features.get('symptom_chest_pain', False) and (features['heart_rate'] > 100 or features['comorbidity_hypertension'] or features['comorbidity_diabetes']):
        diagnosis = {
            'code': 'I21.9',
            'name': 'Acute myocardial infarction',
            'meds': ['Aspirin 300mg', 'Clopidogrel 75mg', 'IV Fluids'],
            'ward': 'ICU',
            'stay': 7
        }
    # 4. Respiratory infections
    elif features['symptom_cough'] and features['symptom_fever'] and features['lab_wbc'] > 15:
        diagnosis = {
            'code': 'J18.9',
            'name': 'Pneumonia, unspecified',
            'meds': ['Amoxicillin 500mg TDS', 'Azithromycin 250mg OD'],
            'ward': 'General',
            'stay': 5
        }
    # 5. Chronic conditions requiring monitoring (elevated labs)
    elif features['comorbidity_hypertension'] and (features['blood_pressure_sys'] > 160 or features['lab_crp'] > 10):
        diagnosis = {
            'code': 'I10',
            'name': 'Essential hypertension (uncontrolled)',
            'meds': ['Amlodipine 5mg OD', 'Lisinopril 10mg OD'],
            'ward': 'General',
            'stay': 2
        }
    elif features['comorbidity_diabetes'] and (features['lab_wbc'] > 11 or features['lab_crp'] > 10):
        diagnosis = {
            'code': 'E11.9',
            'name': 'Type 2 diabetes (uncontrolled)',
            'meds': ['Metformin 500mg BD', 'Insulin if needed'],
            'ward': 'General',
            'stay': 2
        }
    # 6. Stable chronic conditions (outpatient)
    elif features['comorbidity_hypertension']:
        diagnosis = {
            'code': 'I10',
            'name': 'Essential hypertension',
            'meds': ['Amlodipine 5mg OD', 'Lisinopril 10mg OD'],
            'ward': 'Outpatient',
            'stay': 0
        }
    elif features['comorbidity_diabetes']:
        diagnosis = {
            'code': 'E11.9',
            'name': 'Type 2 diabetes',
            'meds': ['Metformin 500mg BD'],
            'ward': 'Outpatient',
            'stay': 0
        }
    # 7. Default (healthy/routine)
    else:
        diagnosis = {
            'code': 'Z00.0',
            'name': 'General medical examination',
            'meds': ['Routine follow-up'],
            'ward': 'Outpatient',
            'stay': 0
        }

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


def analyze_with_chutes(features: dict) -> dict:
    """Analyze patient features using Chutes AI with retry and logging"""
    if st.session_state.get('ai_unavailable', False):
        return fallback_analysis(features)

    clinical_note = build_clinical_note(features)
    from ai_engine import analyze_text_with_chutes

    raw_log = os.path.join(DATA_DIR, "chutes_raw_responses.log")
    attempts = 2

    for attempt in range(1, attempts + 1):
        try:
            start = time.time()
            with st.spinner(f"{CHUTES_MODEL} analyzing clinical presentation... (timeout {CHUTES_TIMEOUT}s) [attempt {attempt}/{attempts}]"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(analyze_text_with_chutes, clinical_note)
                    ai_result = future.result(timeout=CHUTES_TIMEOUT)

            elapsed = time.time() - start

            # Log raw response
            try:
                with open(raw_log, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now().isoformat()} | attempt={attempt} | elapsed={elapsed:.2f}s | raw={repr(ai_result)[:4000]}\n")
            except Exception:
                pass

            # Validate result
            if not ai_result or not isinstance(ai_result, dict) or not ai_result.get("icd10_code"):
                st.warning("AI returned unexpected result — using fallback rules")
                return fallback_analysis(features)

            # Check for inappropriate diagnosis based on severity
            severity_score = features['severity_score']
            icd_code = ai_result.get('icd10_code', '')
            is_inpatient = ai_result.get('inpatient', False)
            
            # Reject chronic/outpatient diagnoses for critical patients
            chronic_codes = ['I10', 'E11', 'E10', 'I15', 'E78']  # Hypertension, Diabetes, Hyperlipidemia
            if severity_score >= 15 and (icd_code in chronic_codes or not is_inpatient):
                st.warning(f"AI diagnosed '{icd_code}' but severity is {severity_score}/25 (Critical/Severe). Using fallback rules.")
                return fallback_analysis(features)
            
            # Verify ICD-10 code
            try:
                icd_info = lookup_icd10(ai_result['icd10_code'])
                if icd_info and icd_info.get('title'):
                    ai_result['diagnosis_name'] = icd_info.get('title')
                    st.success(f"ICD-10 Code Validated: {ai_result['icd10_code']} (took {elapsed:.1f}s)")
            except Exception as e:
                st.warning(f"ICD-10 verification issue: {str(e)}")

            return ai_result

        except concurrent.futures.TimeoutError:
            msg = f"Chutes AI call timed out after {CHUTES_TIMEOUT}s (attempt {attempt}/{attempts})"
            st.warning(msg)
            with open(raw_log, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now().isoformat()} | TIMEOUT | attempt={attempt}\n")
            if attempt < attempts:
                time.sleep(1)
                continue
            st.error(msg + " — switching to fallback")
            st.session_state['ai_unavailable'] = True
            return fallback_analysis(features)

        except Exception as e:
            err = str(e)
            try:
                with open(raw_log, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now().isoformat()} | ERROR | attempt={attempt} | err={err[:2000]}\n")
            except Exception:
                pass
            if "token" in err.lower() or "connection" in err.lower():
                st.session_state['ai_unavailable'] = True
                st.sidebar.error("Chutes AI token/connection issue — switched to fallback rules.")
            if attempt < attempts:
                time.sleep(1)
                continue
            return fallback_analysis(features)

    return fallback_analysis(features)
