import os
from dotenv import load_dotenv
import json
import ollama
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "meditron:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = """
You are a clinical reasoning assistant specialized in medical diagnosis and treatment recommendations. 
Input: free-text medical record (admission notes, discharge summary, labs, vitals).
Output: JSON ONLY (strict), with the following keys:

{
  "icd10_code": "<preferred ICD-10 code (e.g. I50.9) or null>",
  "diagnosis_name": "<short diagnosis name>",
  "confidence": "<0-1 float>",
  "inpatient": true|false,
  "estimated_stay_days": <integer or null>,
  "ward_type": "<ICU|Cardiac|Neurological|General|Isolation|Outpatient>",
  "recommended_medicines": ["list of medicines (dose if obvious)"],
  "extracted": {
     "age": <int|null>,
     "sex": "M|F|Other|null",
     "blood_pressure_sys": <int|null>,
     "blood_pressure_dia": <int|null>,
     "heart_rate": <int|null>,
     "temperature_c": <float|null>,
     "wbc": <float|null>,
     "crp": <float|null>,
     "symptoms": ["list of extracted symptoms"],
     "comorbidities": ["list of comorbidities"]
  },
  "rationale": "<short plain-text explanation (max 120 words)>"
}

- If a field is unknown, set it to null (not empty string).
- Try to return the most precise ICD-10 code available.
- Keep 'recommended_medicines' conservative and list common medicine names; do not invent novel drugs.
- Output valid JSON only; do NOT prepend commentary or add text outside the JSON structure.
"""

def call_ollama_chat(prompt_user: str, system_prompt: str = SYSTEM_PROMPT, model: str = OLLAMA_MODEL, temperature: float = 0.3) -> Optional[Dict[str, Any]]:
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
            format="json",  # Request JSON output format
            options={
                "temperature": temperature,
                "num_ctx": 4096
            }
        )
        
        content = response["message"]["content"]
        
        # Parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON if the model didn't format perfectly
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception as e:
                    raise RuntimeError(f"Failed to parse JSON from model output: {e}\nRaw:\n{content}")
            else:
                raise RuntimeError(f"Model output is not valid JSON:\n{content}")
            
    except Exception as e:
        # Log the error but don't crash the application
        print(f"Ollama inference error: {str(e)}")
        return None

def analyze_text_with_ollama(text: str) -> Dict[str, Any]:
    """
    High-level wrapper that calls the model and ensures all expected keys exist.
    """
    res = call_ollama_chat(text)
    
    if res is None:
        # Fallback to rule-based system when Ollama fails
        return fallback_analysis(text)
    
    # Ensure the result has all required keys with default values
    return normalize_ai_response(res)

def fallback_analysis(text: str) -> Dict[str, Any]:
    """
    Rule-based fallback analysis when AI is unavailable
    """
    # Simple fallback - extract basic info from text
    import re
    
    # Look for fever mentions
    has_fever = bool(re.search(r'fever|temperature|temp|pyrexia', text.lower()))
    symptoms = ["fever"] if has_fever else ["unspecified symptoms"]
    
    return {
        "icd10_code": "R50.9",
        "diagnosis_name": "Fever of unknown origin",
        "confidence": 0.5,
        "inpatient": False,
        "estimated_stay_days": None,
        "ward_type": "Outpatient",
        "recommended_medicines": ["Paracetamol 500mg"],
        "extracted": {
            "age": None,
            "sex": None,
            "blood_pressure_sys": None,
            "blood_pressure_dia": None,
            "heart_rate": None,
            "temperature_c": None if not has_fever else 38.5,
            "wbc": None,
            "crp": None,
            "symptoms": symptoms,
            "comorbidities": []
        },
        "rationale": "Fallback rule-based analysis due to AI unavailability"
    }

def normalize_ai_response(response: Dict) -> Dict[str, Any]:
    """
    Ensure the AI response has all required keys with appropriate defaults
    """
    # Main keys
    keys = ["icd10_code", "diagnosis_name", "confidence", "inpatient", 
            "estimated_stay_days", "ward_type", "recommended_medicines", 
            "extracted", "rationale"]
    
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = response.get(k)
    
    # Handle nulls and types
    if out.get("confidence") is None:
        out["confidence"] = 0.8  # Default confidence
    
    if out.get("recommended_medicines") is None:
        out["recommended_medicines"] = ["Supportive care"]
    
    if out.get("rationale") is None:
        out["rationale"] = "No detailed rationale available"
    
    # extracted subkeys - ensure extracted is a dict before populating
    extracted_defaults = {
        "age": None, "sex": None, "blood_pressure_sys": None, "blood_pressure_dia": None,
        "heart_rate": None, "temperature_c": None, "wbc": None, "crp": None,
        "symptoms": [], "comorbidities": []
    }
    
    extracted = response.get("extracted") or {}
    if not isinstance(extracted, dict):
        extracted = {}
    
    # ensure out["extracted"] is a dict we can safely populate
    out["extracted"] = {}
    for k, v in extracted_defaults.items():
        out["extracted"][k] = extracted.get(k, v)
    
    # Type safety for lists
    if not isinstance(out["extracted"]["symptoms"], list):
        out["extracted"]["symptoms"] = []
    if not isinstance(out["extracted"]["comorbidities"], list):
        out["extracted"]["comorbidities"] = []
    if not isinstance(out["recommended_medicines"], list):
        out["recommended_medicines"] = ["Supportive care"]
    
    return out