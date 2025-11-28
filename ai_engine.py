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

SYSTEM_PROMPT = """
You are a clinical reasoning assistant specialized in medical diagnosis and treatment recommendations. 
You MUST use ICD-10 codes for all diagnoses.

Input: patient clinical data including symptoms, vitals, lab results, and comorbidities.
Output: JSON ONLY (strict), with the following keys:

{
  "icd10_code": "<primary ICD-10 code based on diagnosis>",
  "diagnosis_name": "<official diagnosis name matching ICD-10>",
  "confidence": "<0-1 float>",
  "inpatient": true|false,
  "estimated_stay_days": <integer or null>,
  "ward_type": "<ICU|Cardiac|Neurological|General|Isolation|Outpatient>",
  "recommended_medicines": ["list of standard medications with dosages"],
  "extracted": {
     "age": <int|null>,
     "sex": "M|F|Other|null",
     "blood_pressure_sys": <int|null>,
     "blood_pressure_dia": <int|null>,
     "heart_rate": <int|null>,
     "temperature_c": <float|null>,
     "wbc": <float|null>,
     "crp": <float|null>,
     "symptoms": ["list of identified symptoms"],
     "comorbidities": ["list of identified comorbidities"]
  },
  "rationale": "<clinical reasoning explaining diagnosis and recommendations>"
}

CRITICAL INSTRUCTIONS:
- Use ONLY valid ICD-10 codes from official classification
- Diagnosis name MUST match the ICD-10 code description
- For inpatient=true, provide estimated_stay_days and appropriate ward_type
- Recommended medicines should be standard, evidence-based treatments
- Extract all available clinical data from the input
- Output valid JSON only - no additional text
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
    res = call_ollama_chat(text)
    
    if res is None:
        return fallback_analysis(text)
    
    return normalize_ai_response(res)

def fallback_analysis(text: str) -> Dict[str, Any]:
    """
    Rule-based fallback analysis when AI is unavailable
    """
    # Extract basic info from text using regex patterns
    import re
    
    # Look for common patterns in clinical text
    symptoms = []
    if re.search(r'\bfever\b|\btemperature\b|\btemp\b|\bpyrexia\b', text, re.IGNORECASE):
        symptoms.append("fever")
    if re.search(r'\bcough\b|\bcoughing\b', text, re.IGNORECASE):
        symptoms.append("cough")
    if re.search(r'\bbreath\b|\bbreathing\b|\bdyspnea\b|\bshortness of breath\b', text, re.IGNORECASE):
        symptoms.append("breathlessness")
    if re.search(r'\bhypertension\b|\bhigh blood pressure\b|\bhtn\b', text, re.IGNORECASE):
        symptoms.append("hypertension")
    if re.search(r'\bdiabet\b|\bsugar\b|\bglucose\b', text, re.IGNORECASE):
        symptoms.append("diabetes")
    
    # Simple diagnosis logic based on symptoms
    if "cough" in symptoms and "fever" in symptoms:
        diagnosis = {"code": "J18.9", "name": "Pneumonia, unspecified", "meds": ["Amoxicillin 500mg TDS", "Paracetamol 500mg PRN"]}
        inpatient = True
        stay_days = 5
        ward = "General"
    elif "hypertension" in symptoms:
        diagnosis = {"code": "I10", "name": "Essential (primary) hypertension", "meds": ["Amlodipine 5mg OD", "Lisinopril 10mg OD"]}
        inpatient = False
        stay_days = None
        ward = "Outpatient"
    elif "diabetes" in symptoms:
        diagnosis = {"code": "E11.9", "name": "Type 2 diabetes mellitus without complications", "meds": ["Metformin 500mg BD", "Glucose monitoring"]}
        inpatient = False
        stay_days = None
        ward = "Outpatient"
    else:
        diagnosis = {"code": "R50.9", "name": "Fever, unspecified", "meds": ["Paracetamol 500mg QDS"]}
        inpatient = len(symptoms) > 2
        stay_days = 3 if inpatient else None
        ward = "General" if inpatient else "Outpatient"
    
    return {
        "icd10_code": diagnosis["code"],
        "diagnosis_name": diagnosis["name"],
        "confidence": 0.6,
        "inpatient": inpatient,
        "estimated_stay_days": stay_days,
        "ward_type": ward,
        "recommended_medicines": diagnosis["meds"],
        "extracted": {
            "age": None,
            "sex": None,
            "blood_pressure_sys": None,
            "blood_pressure_dia": None,
            "heart_rate": None,
            "temperature_c": None,
            "wbc": None,
            "crp": None,
            "symptoms": symptoms,
            "comorbidities": []
        },
        "rationale": "Fallback rule-based analysis based on detected symptoms: " + ", ".join(symptoms)
    }

def normalize_ai_response(response: Dict) -> Dict[str, Any]:
    """
    Ensure the AI response has all required keys with appropriate defaults
    """
    # Main keys with defaults
    out = {
        "icd10_code": response.get("icd10_code"),
        "diagnosis_name": response.get("diagnosis_name"),
        "confidence": response.get("confidence", 0.7),
        "inpatient": response.get("inpatient", False),
        "estimated_stay_days": response.get("estimated_stay_days"),
        "ward_type": response.get("ward_type", "General"),
        "recommended_medicines": response.get("recommended_medicines", ["Supportive care"]),
        "rationale": response.get("rationale", "Clinical assessment completed")
    }
    
    # Handle extracted data
    extracted = response.get("extracted", {})
    if not isinstance(extracted, dict):
        extracted = {}
    
    out["extracted"] = {
        "age": extracted.get("age"),
        "sex": extracted.get("sex"),
        "blood_pressure_sys": extracted.get("blood_pressure_sys"),
        "blood_pressure_dia": extracted.get("blood_pressure_dia"),
        "heart_rate": extracted.get("heart_rate"),
        "temperature_c": extracted.get("temperature_c"),
        "wbc": extracted.get("wbc"),
        "crp": extracted.get("crp"),
        "symptoms": extracted.get("symptoms", []),
        "comorbidities": extracted.get("comorbidities", [])
    }
    
    # Ensure lists are actually lists
    if not isinstance(out["extracted"]["symptoms"], list):
        out["extracted"]["symptoms"] = []
    if not isinstance(out["extracted"]["comorbidities"], list):
        out["extracted"]["comorbidities"] = []
    if not isinstance(out["recommended_medicines"], list):
        out["recommended_medicines"] = ["Supportive care"]
    
    # Validate ICD-10 code format
    if out["icd10_code"] and not re.match(r'^[A-Z][0-9]{2}(\.[0-9]{1,})?$', out["icd10_code"]):
        # If ICD-10 code is malformed, set to null
        out["icd10_code"] = None
    
    return out