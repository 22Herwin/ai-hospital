# AI Hospital Management System

A fully interactive **clinical decision support system** using:

- **Chutes AI (Gemma-3 / DeepSeek-R1)** for diagnosis reasoning
- Deterministic **fallback rule engine**
- **ICD-10 validation** via WHO + ICD10API
- **Hospital bed management** with auto-routing
- **Medicine stock system**
- **SQLite local database** for admission logging
- **Admission workflow & logging**
- **Plotly dashboards** for occupancy analysis

---

## Architecture

**Version 2.0** - Completely refactored to modular architecture!

The application is now organized into clean, maintainable modules:

- **`app.py`** - Main orchestrator, imports and coordinates all modules
- **`modules/config.py`** - All configuration, constants, and environment variables
- **`modules/data_manager.py`** - Stock & hospital data operations (CRUD)
- **`modules/clinical_analysis.py`** - AI diagnosis, severity scoring, fallback rules
- **`modules/ui_components.py`** - Streamlit UI components (sidebar, forms, display)
- **`modules/visualizations.py`** - All Plotly charts and dashboards
- **`modules/admission.py`** - Complete admission workflow & timeline

---

## Features

### **AI-Powered Diagnosis**

- AI predicts ICD-10, inpatient/outpatient, meds, rationale, confidence
- ICD-10 lookup enriches/validates diagnosis (title + definition)
- Optional local fallback to **Ollama** if Chutes fails

---

### **Deterministic Fallback Engine**

- If AI token missing / API error/timeout → rule engine returns safe decision

---

### **Hospital Bed Management**

- Editable occupancy per hospital/ward
- Auto-routing: if preferred hospital/ward is full, route to the next with capacity
- Post-admission alerts when capacity ≥85%

---

### **Medicine Stock System**

- Loads and persists stock
- Deducts on admission, replenishment actions

---

### **SQLite Local Database**

- Locally persists all admission records in SQLite
- Real-time sync with **JetBrains DataGrip** for visual inspection
- All admission data stored with patient, diagnosis, ward, hospital, meds, severity, and timestamps

---

### **Admission Logging**

- SQLite: `admissions.db` with structured admission records
- CSV: `data/admission_log.csv` with 11 columns (optional backup)
- Includes patient, diagnosis, ward/hospital, meds, severity, timestamp

---

### **Visual Dashboards**

- Plotly bar/stacked bar/heatmap/gauges for ward/hospital occupancy
- Timeline view of all admissions with filtering by date, ward type, hospital

---

## Project Structure

```
project/
├── app.py                      # Main Streamlit app (orchestrator)
├── app_backup_old.py           # Backup of original monolithic app
├── modules/                    # Modular components
│   ├── __init__.py
│   ├── config.py               # Configuration & constants
│   ├── data_manager.py         # Stock & hospital CRUD operations
│   ├── clinical_analysis.py    # AI diagnosis & severity scoring
│   ├── ui_components.py        # Sidebar, forms, display functions
│   ├── visualizations.py       # Plotly charts & dashboards
│   └── admission.py            # Admission workflow & timeline
├── ai_engine.py                # LLM calls (Chutes + Ollama fallback)
├── icd10_loader.py             # WHO + ICD10API ICD-10 validation
├── sqlite_client.py            # SQLite database operations (CRUD)
├── supabase_client.py          # Optional Supabase backup sync
├── admissions.db               # SQLite local database (auto-created)
├── data/
│   ├── hospitals.csv
│   ├── medicine_stock.csv
│   └── .patient_counter        # Sequential patient ID counter
├── .env
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.8–3.11
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Installation

### 1. Clone the Repository

**HTTPS:**

```bash
git clone https://github.com/22Herwin/ai-hospital.git
cd ai-hospital
```

**SSH:**

```bash
git clone git@github.com:22Herwin/ai-hospital.git
cd ai-hospital
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit>=1.20.0 python-dotenv>=0.21.0 pandas>=1.5.0 numpy>=1.23.0 scikit-learn>=1.2.0 joblib>=1.2.0 plotly>=5.13.0 requests>=2.28.0 python-dateutil>=2.8.2 supabase>=2.0.0 postgrest-py>=0.13.0
```

---

## Environment Variables (`.env`)

Create a `.env` file in the project root:

```bash
# Chutes AI (cloud LLM)
CHUTES_API_TOKEN=your_chutes_api_key
CHUTES_MODEL=unsloth/gemma-3-12b-it
CHUTES_API_URL=https://llm.chutes.ai/v1/chat/completions

# WHO ICD-10 Validation (optional)
WHO_CLIENT_ID=your_who_key
WHO_CLIENT_SECRET=your_who_secret

# Ollama (local fallback LLM)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_TIMEOUT=30

# SQLite Local Database
SQLITE_DB=admissions.db # Your local saved DB
```

If `CHUTES_API_TOKEN` is missing → app auto-falls back to local Ollama (if running).

---

## Running the Application

### Start Streamlit

```bash
streamlit run app.py
```

Open your browser: **http://localhost:8501**

---

## SQLite Database Setup

### Using JetBrains DataGrip (Recommended)

1. **Open DataGrip**
2. **File → New → Data Source → SQLite**
3. **File path:** `yourp-path\admissions.db`
4. **Test Connection** → Click **OK**
5. Right-click database → **Open SQL** to query admissions

### Direct SQLite Query

```bash
sqlite3 admissions.db
sqlite> SELECT * FROM admissions;
```

---

## Auto-Created Files

If missing, the app will create:

- `admissions.db` (SQLite database with admissions table)
- `data/medicine_stock.csv`
- `data/hospitals.csv`
- `data/admission_log.csv`

---

## Usage Notes

- **Confirm Admission** updates hospital occupancy, saves to SQLite, and logs to CSV
- **Auto-routing** chooses the next hospital with available beds in the same ward type
- **Bed Occupancy Editor** lets you simulate capacity scenarios
- **Plotly Charts** visualize ward/hospital capacity (bar/stacked/heatmap/gauges)
- **Timeline View** shows all admissions with filtering by date, ward, and hospital
- **Real-time Sync** — Refresh DataGrip to see new admissions instantly

---

## Production Notes

Demo prototype. For production add:

- Authentication & Role-Based Access Control (RBAC)
- Database encryption & audit logs
- Medical expert validation & bias reviews
- Rate limiting & API security
- Backup & disaster recovery

---

## Troubleshooting

### SQLite Not Showing Data

1. **Check `.env` file** has `SQLITE_DB=admissions.db`
2. **Ensure file path matches** between DataGrip and `.env`
3. **Restart Streamlit** after updating `.env`
4. **Verify database location:** Should be in project root folder

```bash
ls admissions.db  # macOS/Linux
dir admissions.db # Windows
```

### AI Engine Not Working

1. **Check Chutes API Token** in `.env`
2. **App falls back to rule engine** if token missing
3. **Optional:** Run Ollama locally as backup

```bash
ollama pull mistral
ollama serve
```

---

## Developed By — Team 7

- Herwin Dermawan
- M. Dimas Fajar R.
- Chriscinntya Seva Garcia

---

## License

MIT License © 2025 Team 7
