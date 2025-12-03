# AI Hospital Management System

A fully interactive **clinical decision support system** using:

- **Chutes AI (Gemma-3 / DeepSeek-R1)** for diagnosis reasoning
- Deterministic **fallback rule engine**
- **ICD-10 validation** via WHO + ICD10API
- **Hospital bed management** with auto-routing
- **Medicine stock system**
- **Admission workflow & logging**
- **Plotly dashboards** for occupancy analysis

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

### **Admission Logging**

- CSV: `data/admission_log.csv` with 11 columns
- Includes patient, diagnosis, ward/hospital, meds, severity, timestamp

---

### **Visual Dashboards**

- Plotly bar/stacked bar/heatmap/gauges for ward/hospital occupancy

---

## Project Structure

```
project/
├── app.py                 # Streamlit app (UI, workflows, charts)
├── ai_engine.py           # LLM calls (Chutes + Ollama fallback)
├── icd10_loader.py        # WHO + ICD10API ICD-10 validation
├── data/
│   ├── hospitals.csv
│   ├── medicine_stock.csv
│   └── admission_log.csv
├── .env
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10–3.12
- Install dependencies:

```
pip install -r requirements.txt
```

---

## Environment Variables (`.env`)

```
# Chutes (cloud)
CHUTES_API_TOKEN=your_chutes_api_key
CHUTES_MODEL=unsloth/gemma-3-12b-it
CHUTES_API_URL=https://llm.chutes.ai/v1/chat/completions

# WHO ICD (optional)
WHO_CLIENT_ID=your_who_key
WHO_CLIENT_SECRET=your_who_secret

# Ollama (local fallback)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_TIMEOUT=30
```

If `CHUTES_API_TOKEN` is missing or returns 402 → app auto-falls back to local Ollama (if running).

---

## Running the Application

```
streamlit run app.py
```

Open: http://localhost:8501

---

## Auto-Created Files

If missing, the app will create:

- `data/medicine_stock.csv`
- `data/hospitals.csv`
- `data/admission_log.csv`

---

## Usage Notes

- Confirm Admission updates hospital occupancy and logs to CSV.
- Auto-routing chooses the next hospital with available beds in the same ward type.
- Bed occupancy editor lets you simulate capacity scenarios.
- Plotly charts visualize ward/hospital capacity (bar/stacked/heatmap/gauges).

---

## Screenshots

- Clinical Analysis Results
- Ward Capacity Monitor (Plotly)
- Recent Admissions
- Bed Occupancy Editor

---

## Production Notes

Demo prototype. For production add:

- Auth & RBAC, database, rate limiting, audit logs, encryption
- Medical expert validation, bias and safety reviews

---

## Developed By — Team 7

- Herwin Dermawan
- M. Dimas Fajar R.
- Chriscinntya Seva Garcia

## License

MIT License © 2025 Team 7
