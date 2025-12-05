import os
import requests
from typing import Optional, Dict
import collections.abc

# Optional: load .env if present (local dev)
try:
    from dotenv import load_dotenv
    BASE_DIR = os.path.dirname(__file__)
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
except Exception:
    pass

# Streamlit is optional (this module can be used outside Streamlit too)
try:
    import streamlit as st
except Exception:
    st = None

def _find_in_secrets(key: str) -> Optional[str]:
    """Find a key in st.secrets at top-level or any nested section."""
    try:
        if not st or not hasattr(st, "secrets"):
            return None
        sec = st.secrets or {}
        if not sec:
            return None
        # flat
        if key in sec and isinstance(sec[key], str):
            return sec[key]
        # common sections
        for section in ("who", "who_icd", "icd", "chutes"):
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

def get_config(name: str, default: Optional[str] = None) -> Optional[str]:
    """Priority: Streamlit secrets (flat or nested) -> OS env -> default."""
    val = _find_in_secrets(name)
    if not val:
        val = os.getenv(name)
    return val if val is not None else default

# Config (allow overriding via secrets/env)
WHO_CLIENT_ID = get_config("WHO_CLIENT_ID")
WHO_CLIENT_SECRET = get_config("WHO_CLIENT_SECRET")

WHO_TOKEN_URL = get_config("WHO_TOKEN_URL", "https://icdaccessmanagement.who.int/connect/token")
WHO_API_BASE = get_config("WHO_API_BASE", "https://id.who.int/icd/release/10/2019")
ICD10API_FALLBACK = get_config("ICD10API_FALLBACK", "https://icd10api.com/")

# Export to env for downstream libs if not already set
if WHO_CLIENT_ID and not os.getenv("WHO_CLIENT_ID"):
    os.environ["WHO_CLIENT_ID"] = WHO_CLIENT_ID
if WHO_CLIENT_SECRET and not os.getenv("WHO_CLIENT_SECRET"):
    os.environ["WHO_CLIENT_SECRET"] = WHO_CLIENT_SECRET

def get_who_token() -> Optional[str]:
    if not WHO_CLIENT_ID or not WHO_CLIENT_SECRET:
        return None

    token_url = WHO_TOKEN_URL
    if not token_url:
        return None

    data = {
        "grant_type": "client_credentials",
        "scope": "icdapi_access",
        "client_id": WHO_CLIENT_ID,
        "client_secret": WHO_CLIENT_SECRET
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        r = requests.post(token_url, data=data, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        # Use print to avoid strict Streamlit dependency
        print("WHO token error:", e)
        return None

def lookup_code_who(code: str) -> Optional[Dict[str, str]]:
    token = get_who_token()
    if not token:
        return None

    url = f"{WHO_API_BASE}/en/{code}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            j = r.json()
            return {
                "code": code,
                "title": j.get("title", ""),
                "definition": j.get("definition", "")
            }
    except Exception as e:
        print("WHO API error:", e)

    return None

def lookup_code_icd10api(code: str) -> Optional[Dict[str, str]]:
    params = {
        "code": code,
        "r": "json",
        "desc": "long"
    }

    # Ensure ICD10API_FALLBACK is a valid string before using it as URL
    url = ICD10API_FALLBACK
    if not url:
        print("ICD10API_FALLBACK is not configured")
        return None

    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            j = r.json()
            if str(j.get("Valid", "0")) == "1":
                return {
                    "code": code,
                    "title": j.get("ShortDesc", ""),
                    "definition": j.get("LongDesc", "")
                }
    except Exception as e:
        print("ICD10API fallback error:", e)

    return None

def lookup_icd10(code: str) -> Dict[str, str]:
    code = code.upper().strip()

    # Try WHO API first
    res = lookup_code_who(code)
    if res:
        return res

    # Fallback to ICD10API.com
    res = lookup_code_icd10api(code)
    if res:
        return res

    # Last fallback
    return {"code": code, "title": "", "definition": ""}