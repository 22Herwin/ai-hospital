import os
import logging
import collections.abc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Optional: Streamlit secrets (module should work without Streamlit too)
try:
    import streamlit as st
except Exception:
    st = None

def _find_in_secrets(key: str):
    """Find key in st.secrets (flat or nested)."""
    try:
        if not st or not hasattr(st, "secrets"):
            return None
        sec = st.secrets or {}
        if not sec:
            return None
        # flat
        if key in sec:
            return sec[key]
        # nested sections
        for section in ("supabase", "database", "db"):
            if section in sec and isinstance(sec[section], collections.abc.Mapping) and key in sec[section]:
                return sec[section][key]
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

def get_config(name: str, default: str | None = None) -> str | None:
    """Priority: Streamlit secrets (flat/nested) -> OS env -> default."""
    val = _find_in_secrets(name)
    if not val:
        val = os.getenv(name)
    return val if val is not None else default

# Supabase configuration (secrets + env)
SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_KEY")

# Export to env for downstream libs if not already set
if SUPABASE_URL and not os.getenv("SUPABASE_URL"):
    os.environ["SUPABASE_URL"] = SUPABASE_URL
if SUPABASE_KEY and not os.getenv("SUPABASE_KEY"):
    os.environ["SUPABASE_KEY"] = SUPABASE_KEY

def get_supabase_client():
    """Initialize and return Supabase client"""
    try:
        from supabase import create_client
    except ImportError:
        raise ImportError("supabase package not installed. Run: pip install supabase")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL and SUPABASE_KEY not configured")
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in secrets or .env")
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_admission(admission_data: dict) -> bool:
    """Insert admission record into Supabase - matches actual table schema"""
    try:
        client = get_supabase_client()
        
        # Map to ACTUAL Supabase table columns
        record = {
            "patient_id": admission_data.get('patient_id'),
            "admit_time": admission_data.get('admit_time'),
            "ward_type": admission_data.get('ward_type'),
            "estimated_days": admission_data.get('estimated_days'),
            "med_used": admission_data.get('med_used'),
            "qty": admission_data.get('qty'),
            "diagnosis_code": admission_data.get('diagnosis_code'),
            "severity_score": admission_data.get('severity_score')
        }
        
        # Remove None values to avoid schema issues
        record = {k: v for k, v in record.items() if v is not None}
        
        logger.info(f"Inserting admission: {record}")
        response = client.table("admissions").insert(record).execute()
        logger.info(f"Admission inserted: {admission_data.get('patient_id')}")
        return True
    
    except Exception as e:
        logger.error(f"Supabase insert error: {str(e)}", exc_info=True)
        return False

def get_all_admissions() -> list:
    """Fetch all admissions from Supabase"""
    try:
        client = get_supabase_client()
        response = client.table("admissions").select("*").order("admit_time", desc=True).execute()
        return response.data if response.data else []
    
    except Exception as e:
        logger.error(f"Supabase fetch error: {str(e)}")
        return []

def get_patient_admissions(patient_id: str) -> list:
    """Fetch admissions for a specific patient"""
    try:
        client = get_supabase_client()
        response = client.table("admissions").select("*").eq("patient_id", patient_id).order("admit_time", desc=True).execute()
        return response.data if response.data else []
    
    except Exception as e:
        logger.error(f"Supabase patient fetch error: {str(e)}")
        return []

def update_admission(admission_id: int, updates: dict) -> bool:
    """Update an admission record"""
    try:
        client = get_supabase_client()
        response = client.table("admissions").update(updates).eq("id", admission_id).execute()
        logger.info(f"Admission updated: {admission_id}")
        return True
    
    except Exception as e:
        logger.error(f"Supabase update error: {str(e)}")
        return False