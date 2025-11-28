import os
from dotenv import load_dotenv
import requests
from typing import Optional, Dict

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

WHO_CLIENT_ID = os.getenv("WHO_CLIENT_ID")
WHO_CLIENT_SECRET = os.getenv("WHO_CLIENT_SECRET")

WHO_TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
WHO_API_BASE = "https://id.who.int/icd/release/10/2019"

ICD10API_FALLBACK = "https://icd10api.com/"


def get_who_token() -> Optional[str]:
    if not WHO_CLIENT_ID or not WHO_CLIENT_SECRET:
        return None

    data = {
        "grant_type": "client_credentials",
        "scope": "icdapi_access",
        "client_id": WHO_CLIENT_ID,
        "client_secret": WHO_CLIENT_SECRET
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        r = requests.post(WHO_TOKEN_URL, data=data, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
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

    try:
        r = requests.get(ICD10API_FALLBACK, params=params, timeout=15)
        if r.status_code == 200:
            j = r.json()
            if j.get("Valid") == "1":
                return {
                    "code": code,
                    "title": j.get("ShortDesc", ""),
                    "definition": j.get("LongDesc", "")
                }
    except:
        pass

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