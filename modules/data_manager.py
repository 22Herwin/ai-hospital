"""
Data Management - Stock and Hospital Operations
"""
import os
import pandas as pd
import streamlit as st
from typing import Dict, Optional, Any, Tuple
from .config import STOCK_CSV, HOSPITALS_CSV, DATA_DIR, DEFAULT_MEDICINES, DEFAULT_HOSPITALS


@st.cache_resource
def load_stock():
    """Load medicine stock from CSV with fallback initialization"""
    try:
        if os.path.exists(STOCK_CSV):
            df = pd.read_csv(STOCK_CSV)
            df['stock'] = pd.to_numeric(df['stock'], errors='coerce').fillna(0).astype(int)
            if not df.empty:
                return df
        
        # Create default stock if missing or empty
        df = pd.DataFrame({'medicine_name': DEFAULT_MEDICINES, 'stock': [50]*len(DEFAULT_MEDICINES)})
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(STOCK_CSV, index=False)
        st.info("Medicine stock initialized with default inventory")
        return df
        
    except Exception as e:
        st.error(f"Error loading stock: {str(e)}")
        return pd.DataFrame(columns=['medicine_name', 'stock'])


def save_stock(df: pd.DataFrame) -> bool:
    """Save updated stock to CSV"""
    try:
        df.to_csv(STOCK_CSV, index=False)
        load_stock.clear()
        return True
    except Exception as e:
        st.error(f"Error saving stock: {str(e)}")
        return False


@st.cache_resource
def load_hospitals():
    """Load hospitals with ward capacity and occupancy; create defaults if missing."""
    try:
        if os.path.exists(HOSPITALS_CSV):
            df = pd.read_csv(HOSPITALS_CSV)
            # Ensure columns exist and are numeric
            if 'total_beds' in df.columns:
                df['total_beds'] = pd.to_numeric(df['total_beds'], errors='coerce').fillna(0).astype(int)
            else:
                df['total_beds'] = 0
            if 'occupied_beds' in df.columns:
                df['occupied_beds'] = pd.to_numeric(df['occupied_beds'], errors='coerce').fillna(0).astype(int)
            else:
                df['occupied_beds'] = 0
            return df
        else:
            df = pd.DataFrame(DEFAULT_HOSPITALS)
            df.to_csv(HOSPITALS_CSV, index=False)
            return df
    except Exception as e:
        st.error(f"Error loading hospitals: {e}")
        return pd.DataFrame(columns=['hospital_id', 'hospital_name', 'ward_type', 'total_beds', 'occupied_beds'])


def save_hospitals(df: pd.DataFrame) -> bool:
    """Persist hospitals CSV and clear cache."""
    try:
        df.to_csv(HOSPITALS_CSV, index=False)
        load_hospitals.clear()
        return True
    except Exception as e:
        st.error(f"Error saving hospitals: {e}")
        return False


def find_available_hospital(requested_ward: str, hospitals_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Find a hospital ward row with available beds for the requested ward type.
    
    Returns a dict with hospital fields and computed 'available_beds', or None if none found.
    """
    try:
        df = hospitals_df.copy()
        if 'total_beds' not in df.columns or 'occupied_beds' not in df.columns:
            return None
        df['total_beds'] = pd.to_numeric(df['total_beds'], errors='coerce').fillna(0).astype(int)
        df['occupied_beds'] = pd.to_numeric(df['occupied_beds'], errors='coerce').fillna(0).astype(int)

        # Filter by requested ward type
        candidates = df[df['ward_type'].str.lower() == str(requested_ward).lower()].copy()
        if candidates.empty:
            return None

        # Compute available beds
        candidates['available_beds'] = candidates['total_beds'] - candidates['occupied_beds']
        candidates = candidates[candidates['available_beds'] > 0]

        if candidates.empty:
            return None

        # Prefer hospital with most available beds
        candidates['occupancy_pct'] = candidates.apply(
            lambda r: (r['occupied_beds'] / r['total_beds']) if r['total_beds'] > 0 else 1.0,
            axis=1
        )
        candidates = candidates.sort_values(by=['available_beds', 'occupancy_pct'], ascending=[False, True])

        row = candidates.iloc[0]
        return {
            'hospital_id': row.get('hospital_id'),
            'hospital_name': row.get('hospital_name'),
            'ward_type': row.get('ward_type'),
            'total_beds': int(row.get('total_beds', 0)),
            'occupied_beds': int(row.get('occupied_beds', 0)),
            'available_beds': int(row.get('available_beds', 0))
        }
    except Exception:
        return None


def check_ward_capacity_and_alert(requested_ward: str, hospitals_df: pd.DataFrame) -> Tuple[bool, str]:
    """Check aggregate capacity for a ward type and return (has_capacity, message)."""
    try:
        df = hospitals_df.copy()
        if 'total_beds' not in df.columns or 'occupied_beds' not in df.columns:
            return False, "Hospital data missing bed information."

        df['total_beds'] = pd.to_numeric(df['total_beds'], errors='coerce').fillna(0).astype(int)
        df['occupied_beds'] = pd.to_numeric(df['occupied_beds'], errors='coerce').fillna(0).astype(int)

        matching = df[df['ward_type'].str.lower() == str(requested_ward).lower()]
        if matching.empty:
            return False, f"No wards of type '{requested_ward}' found in hospital list."

        total_beds = int(matching['total_beds'].sum())
        total_occupied = int(matching['occupied_beds'].sum())
        total_available = total_beds - total_occupied
        occupancy_pct = int((total_occupied / total_beds) * 100) if total_beds > 0 else 100

        if total_available > 0:
            msg = f"{total_available} bed(s) available across {matching['hospital_name'].nunique()} hospital(s) for '{requested_ward}' ({occupancy_pct}% occupied)."
            return True, msg
        else:
            if occupancy_pct >= 95:
                note = "CRITICAL - no beds, consider overflow/transfer."
            elif occupancy_pct >= 85:
                note = "WARNING - capacity critically low."
            else:
                note = "No beds available at the moment."
            msg = f"No available beds for '{requested_ward}' ({occupancy_pct}% occupied). {note}"
            return False, msg
    except Exception as e:
        return False, f"Capacity check error: {str(e)}"


def generate_sequential_patient_id() -> str:
    """Generate sequential patient ID by checking database for highest existing ID"""
    import datetime
    import re
    import sys
    
    try:
        # Try to get the highest patient ID from database
        try:
            # Import at runtime to avoid circular dependencies
            import sqlite_client
            admissions = sqlite_client.get_all_admissions()
            
            if admissions and len(admissions) > 0:
                # Extract numeric parts from all patient IDs
                max_num = 0
                for admission in admissions:
                    pid = admission.get('patient_id', '')
                    match = re.match(r'P(\d+)', pid)
                    if match:
                        num = int(match.group(1))
                        max_num = max(max_num, num)
                
                # Next ID is max + 1
                if max_num > 0:
                    next_id = max_num + 1
                    return f"P{next_id:06d}"
        except Exception as db_err:
            # Database query failed, use counter file
            pass
        
        # Fallback to counter file
        counter_file = os.path.join(DATA_DIR, '.patient_counter')
        
        if os.path.exists(counter_file):
            with open(counter_file, 'r') as f:
                count = int(f.read().strip())
        else:
            count = 0
        
        count += 1
        with open(counter_file, 'w') as f:
            f.write(str(count))
        
        return f"P{count:06d}"
    except Exception as e:
        # Final fallback: timestamp-based
        import datetime
        return f"P{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
