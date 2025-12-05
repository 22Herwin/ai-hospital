import os
from dotenv import load_dotenv
import json
import requests
from typing import Dict, Any, Optional
import re
import collections.abc

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---- Secrets + Env helper ----
try:
    import streamlit as st
except Exception:
    st = None

def _find_in_secrets(key: str) -> Optional[str]:
    """Find key in st.secrets at top-level or any nested section."""
    try:
        if not st or not hasattr(st, "secrets"):
            return None
        sec = st.secrets or {}
        if not sec:
            return None
        # flat
        if key in sec:
            val = sec[key]
            if isinstance(val, str):
                return val
            # Try to serialize non-string secret values to a string representation
            try:
                return json.dumps(val)
            except Exception:
                return str(val)
        # common nested sections
        for section in ("chutes", "deepseek", "ollama", "supabase"):
            if section in sec and isinstance(sec[section], collections.abc.Mapping) and key in sec[section]:
                val = sec[section][key]
                if isinstance(val, str):
                    return val
                try:
                    return json.dumps(val)
                except Exception:
                    return str(val)
        # deep search
        def dfs(d):
            for _, v in d.items():
                if isinstance(v, collections.abc.Mapping):
                    if key in v:
                        return v[key]
                    r = dfs(v)
                    if r is not None:
                        return r
            return None
        return dfs(sec)
    except Exception:
        return None

def get_config(name: str, default: Optional[str] = None) -> Optional[str]:
    """Priority: Streamlit secrets (flat or nested) -> OS env -> default."""
    val = _find_in_secrets(name)
    if not val:
        val = os.getenv(name)
    return val if val is not None else default

# ---- Chutes AI configuration (secrets + env) ----
CHUTES_API_URL = get_config("CHUTES_API_URL", "https://llm.chutes.ai/v1/chat/completions")
CHUTES_API_TOKEN = get_config("CHUTES_API_TOKEN", None)
CHUTES_MODEL = get_config("CHUTES_MODEL", "unsloth/gemma-3-12b-it")

# Export to env for downstream libs (don’t override if already set)
if CHUTES_API_URL and not os.getenv("CHUTES_API_URL"):
    os.environ["CHUTES_API_URL"] = CHUTES_API_URL
if CHUTES_API_TOKEN and not os.getenv("CHUTES_API_TOKEN"):
    os.environ["CHUTES_API_TOKEN"] = CHUTES_API_TOKEN
if CHUTES_MODEL and not os.getenv("CHUTES_MODEL"):
    os.environ["CHUTES_MODEL"] = CHUTES_MODEL

# Define specific disease categories with STRICT criteria
HOSPITALIZATION_DISEASES = {
    "J18.9": {
        "name": "Pneumonia, unspecified",
        "ward": "General",
        "stay_days": 5,
        "meds": ["Amoxicillin 500mg TDS", "Azithromycin 250mg OD", "Paracetamol 500mg PRN"],
        "criteria": "fever + cough + abnormal lung sounds + elevated WBC/CRP"
    },
    "I21.9": {
        "name": "Acute myocardial infarction, unspecified",
        "ward": "ICU",
        "stay_days": 7,
        "meds": ["Aspirin 300mg", "Clopidogrel 75mg", "Atorvastatin 80mg"],
        "criteria": "chest pain + ECG changes + elevated cardiac enzymes"
    },
    "I63.9": {
        "name": "Cerebral infarction, unspecified",
        "ward": "Neurological",
        "stay_days": 10,
        "meds": ["Aspirin 100mg", "Atorvastatin 40mg", "Mannitol IV PRN"],
        "criteria": "neurological deficits + imaging confirmation"
    },
    "A41.9": {
        "name": "Sepsis, unspecified",
        "ward": "ICU",
        "stay_days": 14,
        "meds": ["Meropenem 1g IV", "Vancomycin 1g IV", "IV Fluids"],
        "criteria": "fever + elevated WBC/CRP + systemic symptoms"
    },
    "J15.9": {
        "name": "Bacterial pneumonia, unspecified",
        "ward": "Isolation",
        "stay_days": 7,
        "meds": ["Ceftriaxone 1g IV", "Azithromycin 500mg IV", "Oxygen therapy"],
        "criteria": "fever + cough + purulent sputum + elevated inflammatory markers"
    }
}

OUTPATIENT_DISEASES = {
    "I10": {
        "name": "Essential (primary) hypertension",
        "ward": "Outpatient",
        "stay_days": 0,
        "meds": ["Amlodipine 5mg OD", "Lisinopril 10mg OD"],
        "criteria": "elevated BP without acute complications"
    },
    "E11.9": {
        "name": "Type 2 diabetes mellitus without complications",
        "ward": "Outpatient",
        "stay_days": 0,
        "meds": ["Metformin 500mg BD", "Glucose monitoring"],
        "criteria": "elevated glucose without acute complications"
    },
    "J06.9": {
        "name": "Acute upper respiratory infection, unspecified",
        "ward": "Outpatient",
        "stay_days": 0,
        "meds": ["Paracetamol 500mg QDS", "Chlorpheniramine 4mg TDS"],
        "criteria": "cough/cold symptoms without systemic illness"
    },
    "K29.70": {
        "name": "Gastritis, unspecified, without bleeding",
        "ward": "Outpatient",
        "stay_days": 0,
        "meds": ["Omeprazole 20mg OD", "Antacids PRN"],
        "criteria": "dyspepsia without alarm features"
    },
    "M54.50": {
        "name": "Low back pain, unspecified",
        "ward": "Outpatient",
        "stay_days": 0,
        "meds": ["Ibuprofen 400mg TDS", "Muscle relaxants PRN"],
        "criteria": "mechanical back pain without neurological deficits"
    }
}

HEALTHY_CODE = {
    "Z00.0": {
        "name": "General medical examination",
        "ward": "Outpatient",
        "stay_days": 0,
        "meds": ["Routine follow-up"],
        "criteria": "no acute symptoms, normal examination"
    }
}

SYSTEM_PROMPT = """
You are a clinical reasoning assistant specialized in medical diagnosis using ICD-10 codes.

CRITICAL DIAGNOSIS RULES - READ CAREFULLY:

1. **DEFAULT TO HEALTHY**: If no significant symptoms are present AND vitals/labs are normal → Use Z00.0

2. **HOSPITALIZATION REQUIRED** (use ONLY if STRICT criteria met):
   - J18.9 (Pneumonia): MUST have fever >38°C + cough + breathlessness + elevated WBC/CRP
   - I21.9 (MI): MUST have chest pain + abnormal ECG/cardiac markers + risk factors  
   - I63.9 (Stroke): MUST have neurological deficits + imaging findings
   - A41.9 (Sepsis): MUST have fever + systemic symptoms + significantly elevated inflammatory markers
   - J15.9 (Bacterial pneumonia): MUST have fever + purulent sputum + consolidation on imaging

3. **OUTPATIENT MANAGEMENT** (use when mild-moderate symptoms):
   - I10 (Hypertension): Elevated BP without acute complications
   - E11.9 (Diabetes): Elevated glucose without DKA/HHS
   - J06.9 (URI): Cough/cold symptoms without systemic illness
   - K29.70 (Gastritis): Dyspepsia without bleeding/perforation
   - M54.50 (Back pain): Mechanical pain without neurological deficits

4. **HEALTHY EXAMINATION** (Z00.0) when:
   - No fever, no significant symptoms
   - Normal vital signs (BP <140/90, HR 60-100, Temp <37.5°C)
   - Normal labs (WBC 4-11, CRP <10)
   - No acute distress

OUTPUT FORMAT - JSON ONLY:
{
  "icd10_code": "EXACT_ICD10_CODE",
  "diagnosis_name": "MATCHING_NAME", 
  "confidence": 0.0-1.0,
  "inpatient": true|false,
  "estimated_stay_days": number_or_null,
  "ward_type": "ICU|General|Neurological|Isolation|Outpatient",
  "recommended_medicines": ["list", "of", "meds"],
  "rationale": "clinical reasoning matching criteria above"
}

IMPORTANT: Be CONSERVATIVE. Only diagnose serious conditions when clear evidence exists.
"""

def call_chutes_chat(prompt_user: str, system_prompt: Optional[str] = SYSTEM_PROMPT, model: Optional[str] = CHUTES_MODEL, temperature: float = 0.1) -> Optional[Dict[str, Any]]:
    """
    Calls Chutes AI chat completions endpoint via HTTP.
    Returns parsed JSON dict or None on failure.
    """
    if not CHUTES_API_TOKEN:
        # No token configured
        raise RuntimeError("CHUTES_API_TOKEN not set in environment")

    # Ensure we have concrete string values for system_prompt and model to satisfy type checkers
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    if model is None:
        # fallback to the module-level CHUTES_MODEL or a safe literal default
        model = CHUTES_MODEL or "deepseek-ai/DeepSeek-R1"

    headers = {
        "Authorization": f"Bearer {CHUTES_API_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_user}
        ],
        "temperature": temperature,
        # providers often accept top_p/top_k, keep safe defaults omitted here
    }

    try:
        # Ensure CHUTES_API_URL is set (requests.post requires a concrete str/bytes)
        if not CHUTES_API_URL:
            raise RuntimeError("CHUTES_API_URL not set in environment")
        r = requests.post(CHUTES_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        j = r.json()

        # Try to extract message content from common fields
        content = None
        # Common OpenAI-like response structure
        if isinstance(j, dict) and "choices" in j and isinstance(j["choices"], list) and len(j["choices"]) > 0:
            choice = j["choices"][0]
            # try chat message
            if isinstance(choice.get("message"), dict) and choice["message"].get("content"):
                content = choice["message"]["content"]
            # or text field
            elif choice.get("text"):
                content = choice.get("text")
            # or delta/content streaming
            elif choice.get("output"):
                content = choice.get("output")
        # Some providers return top-level 'message' or 'content'
        if content is None:
            if isinstance(j.get("message"), dict) and j["message"].get("content"):
                content = j["message"]["content"]
            elif isinstance(j.get("content"), str):
                content = j.get("content")
            elif isinstance(j.get("output"), str):
                content = j.get("output")
            else:
                # as last resort, stringify entire response
                content = json.dumps(j)

        # Ensure content is a string before calling strip()
        if isinstance(content, str):
            content_clean = content.strip()
        else:
            # If content is None use empty string; if it's another type serialize to JSON,
            # otherwise fall back to Python's str() representation.
            if content is None:
                content_clean = ""
            else:
                try:
                    content_clean = json.dumps(content)
                except Exception:
                    content_clean = str(content)
            content_clean = content_clean.strip()

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
        # Bubble up as None for callers to fallback
        print(f"Chutes inference error: {str(e)}")
        return None


def analyze_text_with_chutes(text: str) -> Dict[str, Any]:
    """
    High-level wrapper that calls the model for clinical analysis using Chutes.
    """
    # Build the final prompt with clinical data
    clinical_prompt = f"""
    Analyze this patient case CONSERVATIVELY. Only diagnose serious conditions when clear evidence exists.

    CLINICAL DATA:
    {text}

    REMEMBER: Default to Z00.0 (healthy examination) if no significant abnormalities.

    Return valid JSON with: icd10_code, diagnosis_name, inpatient, estimated_stay_days, ward_type, recommended_medicines, and rationale.
    """

    res = call_chutes_chat(clinical_prompt)

    if res is None:
        return fallback_analysis(text)

    return normalize_ai_response(res)


def fallback_analysis(text: str) -> Dict[str, Any]:
    """
    Rule-based fallback analysis when AI is unavailable - VERY CONSERVATIVE
    """
    import re

    # Extract symptoms from text
    text_lower = text.lower()

    # Check for specific patterns - be very conservative
    has_fever = re.search(r'\bfever\b|\btemperature\b|\btemp\b|\bpyrexia\b', text_lower)
    has_cough = re.search(r'\bcough\b|\bcoughing\b', text_lower)
    has_breathless = re.search(r'\bbreath\b|\bbreathing\b|\bdyspnea\b|\bshortness of breath\b', text_lower)
    has_chest_pain = re.search(r'\bchest pain\b|\bchest discomfort\b', text_lower)
    has_hypertension = re.search(r'\bhypertension\b|\bhigh blood pressure\b|\bhtn\b', text_lower)
    has_diabetes = re.search(r'\bdiabet\b|\bsugar\b|\bglucose\b', text_lower)
    has_neuro = re.search(r'\bweakness\b|\bnumbness\b|\bparalysis\b|\bstroke\b', text_lower)
    has_sepsis = re.search(r'\bsepsis\b|\bseptic\b|\binfection\b', text_lower)

    # Extract lab values using regex
    wbc_match = re.search(r'WBC[:\s]*([0-9.]+)', text, re.IGNORECASE)
    crp_match = re.search(r'CRP[:\s]*([0-9.]+)', text, re.IGNORECASE)
    temp_match = re.search(r'Temperature[:\s]*([0-9.]+)', text, re.IGNORECASE)
    bp_match = re.search(r'Blood Pressure[:\s]*([0-9]+)/([0-9]+)', text, re.IGNORECASE)

    wbc_value = float(wbc_match.group(1)) if wbc_match else 7.0
    crp_value = float(crp_match.group(1)) if crp_match else 5.0
    temp_value = float(temp_match.group(1)) if temp_match else 36.7
    bp_sys = int(bp_match.group(1)) if bp_match else 120
    bp_dia = int(bp_match.group(2)) if bp_match else 80

    # VERY CONSERVATIVE diagnosis - require clear evidence
    if has_fever and has_cough and has_breathless and wbc_value > 13 and crp_value > 50:
        diagnosis = HOSPITALIZATION_DISEASES["J18.9"]  # Pneumonia - strict criteria
        code = "J18.9"
    elif has_chest_pain and has_hypertension and bp_sys > 180:
        diagnosis = HOSPITALIZATION_DISEASES["I21.9"]  # MI - strict criteria
        code = "I21.9"
    elif has_neuro:
        diagnosis = HOSPITALIZATION_DISEASES["I63.9"]  # Stroke
        code = "I63.9"
    elif has_sepsis and has_fever and crp_value > 100:
        diagnosis = HOSPITALIZATION_DISEASES["A41.9"]  # Sepsis - strict criteria
        code = "A41.9"
    elif has_hypertension and bp_sys > 140:
        diagnosis = OUTPATIENT_DISEASES["I10"]  # Hypertension
        code = "I10"
    elif has_diabetes:
        diagnosis = OUTPATIENT_DISEASES["E11.9"]  # Diabetes
        code = "E11.9"
    elif has_cough or has_fever:
        diagnosis = OUTPATIENT_DISEASES["J06.9"]  # URI
        code = "J06.9"
    else:
        # DEFAULT TO HEALTHY if no clear evidence of disease
        diagnosis = HEALTHY_CODE["Z00.0"]
        code = "Z00.0"

    # Determine if inpatient based on diagnosis category
    is_inpatient = code in HOSPITALIZATION_DISEASES

    return {
        "icd10_code": code,
        "diagnosis_name": diagnosis["name"],
        "confidence": 0.6,
        "inpatient": is_inpatient,
        "estimated_stay_days": diagnosis["stay_days"] if is_inpatient else 0,
        "ward_type": diagnosis["ward"],
        "recommended_medicines": diagnosis["meds"],
        "rationale": f"Conservative fallback analysis: {diagnosis['name']}. Using strict criteria to avoid over-diagnosis."
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
    else:
        # If AI returns an unknown code, default to healthy
        out["icd10_code"] = "Z00.0"
        out["diagnosis_name"] = "General medical examination"
        out["inpatient"] = False
        out["ward_type"] = "Outpatient"
        out["estimated_stay_days"] = 0
        out["recommended_medicines"] = ["Routine follow-up"]
        out["rationale"] = "Unknown diagnosis code - defaulting to healthy examination"

    # Ensure lists are actually lists
    if not isinstance(out["recommended_medicines"], list):
        out["recommended_medicines"] = ["Supportive care"]

    return out