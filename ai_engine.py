import os
from dotenv import load_dotenv
import json
import ollama
from typing import Dict, Any, Optional
import re

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "meditron:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Define specific disease categories
HOSPITALIZATION_DISEASES = {
    "J18.9": {"name": "Pneumonia, unspecified", "ward": "General", "stay_days": 5, "meds": ["Amoxicillin 500mg TDS", "Azithromycin 250mg OD", "Paracetamol 500mg PRN"]},
    "I21.9": {"name": "Acute myocardial infarction, unspecified", "ward": "ICU", "stay_days": 7, "meds": ["Aspirin 300mg", "Clopidogrel 75mg", "Atorvastatin 80mg"]},
    "I63.9": {"name": "Cerebral infarction, unspecified", "ward": "Neurological", "stay_days": 10, "meds": ["Aspirin 100mg", "Atorvastatin 40mg", "Mannitol IV PRN"]},
    "A41.9": {"name": "Sepsis, unspecified", "ward": "ICU", "stay_days": 14, "meds": ["Meropenem 1g IV", "Vancomycin 1g IV", "IV Fluids"]},
    "J15.9": {"name": "Bacterial pneumonia, unspecified", "ward": "Isolation", "stay_days": 7, "meds": ["Ceftriaxone 1g IV", "Azithromycin 500mg IV", "Oxygen therapy"]}
}

OUTPATIENT_DISEASES = {
    "I10": {"name": "Essential (primary) hypertension", "ward": "Outpatient", "stay_days": 0, "meds": ["Amlodipine 5mg OD", "Lisinopril 10mg OD"]},
    "E11.9": {"name": "Type 2 diabetes mellitus without complications", "ward": "Outpatient", "stay_days": 0, "meds": ["Metformin 500mg BD", "Glucose monitoring"]},
    "J06.9": {"name": "Acute upper respiratory infection, unspecified", "ward": "Outpatient", "stay_days": 0, "meds": ["Paracetamol 500mg QDS", "Chlorpheniramine 4mg TDS"]},
    "K29.70": {"name": "Gastritis, unspecified, without bleeding", "ward": "Outpatient", "stay_days": 0, "meds": ["Omeprazole 20mg OD", "Antacids PRN"]},
    "M54.50": {"name": "Low back pain, unspecified", "ward": "Outpatient", "stay_days": 0, "meds": ["Ibuprofen 400mg TDS", "Muscle relaxants PRN"]}
}

HEALTHY_CODE = {
    "Z00.0": {"name": "General medical examination", "ward": "Outpatient", "stay_days": 0, "meds": ["Routine follow-up"]}
}

SYSTEM_PROMPT = """
You are a clinical reasoning assistant specialized in medical diagnosis using ICD-10 codes.

CRITICAL GUIDELINES:
1. You MUST select from these specific ICD-10 codes only:

HOSPITALIZATION REQUIRED (Inpatient):
- J18.9: Pneumonia, unspecified → Ward: General, Stay: 5 days
- I21.9: Acute myocardial infarction → Ward: ICU, Stay: 7 days  
- I63.9: Cerebral infarction → Ward: Neurological, Stay: 10 days
- A41.9: Sepsis, unspecified → Ward: ICU, Stay: 14 days
- J15.9: Bacterial pneumonia → Ward: Isolation, Stay: 7 days

OUTPATIENT MANAGEMENT:
- I10: Essential hypertension → Ward: Outpatient
- E11.9: Type 2 diabetes → Ward: Outpatient
- J06.9: Acute upper respiratory infection → Ward: Outpatient
- K29.70: Gastritis → Ward: Outpatient
- M54.50: Low back pain → Ward: Outpatient

HEALTHY:
- Z00.0: General medical examination → Ward: Outpatient

2. Output JSON ONLY with these exact keys:
{
  "icd10_code": "EXACT_ICD10_CODE_FROM_ABOVE_LIST",
  "diagnosis_name": "MATCHING_DIAGNOSIS_NAME",
  "confidence": 0.0-1.0,
  "inpatient": true|false,
  "estimated_stay_days": number_or_null,
  "ward_type": "ICU|General|Neurological|Isolation|Outpatient",
  "recommended_medicines": ["list", "of", "medicines"],
  "rationale": "clinical reasoning"
}

3. Hospitalization Criteria:
- Admit if: high fever + abnormal labs, chest pain + ECG changes, neurological deficits, sepsis signs, respiratory distress
- Outpatient if: stable vitals, mild symptoms, chronic conditions

4. DO NOT invent new codes. Use ONLY the codes provided above.

Input patient data: {clinical_data}
"""

def call_ollama_chat(prompt_user: str, system_prompt: str = SYSTEM_PROMPT, model: str = OLLAMA_MODEL, temperature: float = 0.1) -> Optional[Dict[str, Any]]:
    """
    Calls Ollama chat completions endpoint for local model inference.
    Returns parsed JSON dict or None on failure.
    """
    try:
        # Configure ollama client
        client = ollama.Client(host=OLLAMA_HOST)
        
        # Make the API call
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user}
            ],
            format="json",
            options={
                "temperature": temperature,
                "num_ctx": 4096,
                "top_k": 40,
                "top_p": 0.9
            }
        )
        
        content = response["message"]["content"]
        
        # Clean and parse JSON response
        content_clean = content.strip()
        
        # Remove any markdown code blocks if present
        if content_clean.startswith('```json'):
            content_clean = content_clean[7:]
        if content_clean.endswith('```'):
            content_clean = content_clean[:-3]
        content_clean = content_clean.strip()
        
        try:
            return json.loads(content_clean)
        except json.JSONDecodeError as e:
            # Try to extract JSON using regex
            json_match = re.search(r'\{[\s\S]*\}', content_clean)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception:
                    raise RuntimeError(f"Failed to parse JSON from model output: {e}\nRaw:\n{content_clean}")
            else:
                raise RuntimeError(f"Model output is not valid JSON:\n{content_clean}")
            
    except Exception as e:
        print(f"Ollama inference error: {str(e)}")
        return None

def analyze_text_with_ollama(text: str) -> Dict[str, Any]:
    """
    High-level wrapper that calls the model for clinical analysis.
    """
    # Build the final prompt with clinical data
    clinical_prompt = f"""
    Analyze this patient case and provide diagnosis using ONLY the specified ICD-10 codes:
    
    CLINICAL DATA:
    {text}
    
    Return valid JSON with icd10_code, diagnosis_name, inpatient, estimated_stay_days, ward_type, recommended_medicines, and rationale.
    """
    
    res = call_ollama_chat(clinical_prompt)
    
    if res is None:
        return fallback_analysis(text)
    
    return normalize_ai_response(res)

def fallback_analysis(text: str) -> Dict[str, Any]:
    """
    Rule-based fallback analysis when AI is unavailable
    """
    import re
    
    # Extract symptoms from text
    text_lower = text.lower()
    symptoms = []
    
    # Check for specific patterns
    has_fever = re.search(r'\bfever\b|\btemperature\b|\btemp\b|\bpyrexia\b', text_lower)
    has_cough = re.search(r'\bcough\b|\bcoughing\b', text_lower)
    has_breathless = re.search(r'\bbreath\b|\bbreathing\b|\bdyspnea\b|\bshortness of breath\b', text_lower)
    has_chest_pain = re.search(r'\bchest pain\b|\bchest discomfort\b', text_lower)
    has_hypertension = re.search(r'\bhypertension\b|\bhigh blood pressure\b|\bhtn\b', text_lower)
    has_diabetes = re.search(r'\bdiabet\b|\bsugar\b|\bglucose\b', text_lower)
    has_neuro = re.search(r'\bweakness\b|\bnumbness\b|\bparalysis\b|\bstroke\b', text_lower)
    has_sepsis = re.search(r'\bsepsis\b|\bseptic\b|\binfection\b', text_lower)
    
    # Rule-based diagnosis
    if has_fever and has_cough and has_breathless:
        diagnosis = HOSPITALIZATION_DISEASES["J18.9"]  # Pneumonia
    elif has_chest_pain and has_hypertension:
        diagnosis = HOSPITALIZATION_DISEASES["I21.9"]  # MI
    elif has_neuro:
        diagnosis = HOSPITALIZATION_DISEASES["I63.9"]  # Stroke
    elif has_sepsis and has_fever:
        diagnosis = HOSPITALIZATION_DISEASES["A41.9"]  # Sepsis
    elif has_hypertension:
        diagnosis = OUTPATIENT_DISEASES["I10"]  # Hypertension
    elif has_diabetes:
        diagnosis = OUTPATIENT_DISEASES["E11.9"]  # Diabetes
    elif has_cough or has_fever:
        diagnosis = OUTPATIENT_DISEASES["J06.9"]  # URI
    else:
        diagnosis = HEALTHY_CODE["Z00.0"]  # Healthy
    
    # Determine if inpatient based on diagnosis category
    is_inpatient = diagnosis["icd10_code"] in HOSPITALIZATION_DISEASES if "icd10_code" in diagnosis else diagnosis in HOSPITALIZATION_DISEASES.values()
    
    return {
        "icd10_code": list(diagnosis.keys())[0] if isinstance(diagnosis, dict) and "icd10_code" not in diagnosis else diagnosis.get("icd10_code", "Z00.0"),
        "diagnosis_name": diagnosis["name"],
        "confidence": 0.6,
        "inpatient": is_inpatient,
        "estimated_stay_days": diagnosis["stay_days"] if is_inpatient else 0,
        "ward_type": diagnosis["ward"],
        "recommended_medicines": diagnosis["meds"],
        "rationale": f"Fallback analysis: {diagnosis['name']} based on symptoms detected"
    }

def normalize_ai_response(response: Dict) -> Dict[str, Any]:
    """
    Ensure the AI response has all required keys with appropriate defaults
    """
    # Main keys with defaults
    out = {
        "icd10_code": response.get("icd10_code"),
        "diagnosis_name": response.get("diagnosis_name"),
        "confidence": min(max(response.get("confidence", 0.7), 0.0), 1.0),  # clamp 0-1
        "inpatient": response.get("inpatient", False),
        "estimated_stay_days": response.get("estimated_stay_days", 0),
        "ward_type": response.get("ward_type", "General"),
        "recommended_medicines": response.get("recommended_medicines", ["Supportive care"]),
        "rationale": response.get("rationale", "Clinical assessment completed")
    }
    
    # Validate ICD-10 code against our defined diseases
    icd_code = out["icd10_code"]
    if icd_code in HOSPITALIZATION_DISEASES:
        # Ensure hospitalization settings match our definitions
        disease_info = HOSPITALIZATION_DISEASES[icd_code]
        out["inpatient"] = True
        out["ward_type"] = disease_info["ward"]
        out["estimated_stay_days"] = disease_info["stay_days"]
        if not out["recommended_medicines"] or out["recommended_medicines"] == ["Supportive care"]:
            out["recommended_medicines"] = disease_info["meds"]
    elif icd_code in OUTPATIENT_DISEASES:
        disease_info = OUTPATIENT_DISEASES[icd_code]
        out["inpatient"] = False
        out["ward_type"] = "Outpatient"
        out["estimated_stay_days"] = 0
        if not out["recommended_medicines"] or out["recommended_medicines"] == ["Supportive care"]:
            out["recommended_medicines"] = disease_info["meds"]
    elif icd_code in HEALTHY_CODE:
        disease_info = HEALTHY_CODE[icd_code]
        out["inpatient"] = False
        out["ward_type"] = "Outpatient"
        out["estimated_stay_days"] = 0
        out["recommended_medicines"] = disease_info["meds"]
    
    # Ensure lists are actually lists
    if not isinstance(out["recommended_medicines"], list):
        out["recommended_medicines"] = ["Supportive care"]
    
    return out