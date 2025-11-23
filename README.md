# Hospital Admission & Medicine Stock Prediction System (Demo)

## Overview

This project is a **demo ML system** that simulates a simplified hospital workflow for patient admission. It includes:

- Diagnosis prediction
- Inpatient admission decision
- Ward assignment
- Estimated length of stay
- Medicine recommendation
- Local medicine stock management

The system uses machine learning models, rule-based logic, and a Streamlit interface.

---

## What This Project Demonstrates

- A full machine‑learning pipeline (dataset → training → deployment)
- Synthetic patient data generation
- Multiple ML models working together
- A functional, interactive hospital AI prototype
- Local stock and admission logging

This is ideal for learning or prototyping healthcare ML systems.

---

## Project Structure

| File / Folder         | Description                                             |
| --------------------- | ------------------------------------------------------- |
| `generate_dataset.py` | Creates synthetic patient CSV data                      |
| `train_models.py`     | Trains all ML models and saves them as `.pkl`           |
| `app.py`              | Streamlit app for interacting with the system           |
| `data/`               | Contains patient dataset, admission log, medicine stock |
| `models/`             | Trained ML models                                       |
| `requirements.txt`    | Python dependencies                                     |

---

## System Architecture

### 1. Dataset Generator — `generate_dataset.py`

- Generates clinical-like patient features
- Applies rules to assign diagnoses
- Saves to `patients_sample.csv`

### 2. Model Training Pipeline — `train_models.py`

Trains:

- Diagnosis model
- Inpatient decision model
- Ward classification model
- Length-of-stay regression model
- Medicine recommendation model

Outputs saved in `/models`.

### 3. Streamlit App — `app.py`

- Loads trained models
- Provides an input UI
- Predicts admission info
- Updates medicine stock
- Logs admissions locally

Includes fallback rules if a model file is missing.

---

## Requirements

- Python **3.8–3.11**

Install dependencies:

```
pip install -r requirements.txt
```

If no requirements.txt:

```
pip install pandas numpy scikit-learn joblib streamlit python-dotenv
```

---

## How to Download the Project

### HTTPS:

```
git clone https://github.com/22Herwin/ml-hospital.git
cd ml-hospital
```

### SSH:

```
git clone git@github.com:22Herwin/ml-hospital.git
```

---

## How to Start the Project

### 1. Create virtual environment (optional)

**Windows:**

```
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux/macOS:**

```
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Generate dataset

```
python generate_dataset.py --n 1000 --out data/patients_sample.csv
```

Optional:

```
python generate_dataset.py --n 1000 --seed 42 --out data/patients_sample.csv
```

### 4. Train models

```
python train_models.py --data data/patients_sample.csv --out_dir models
```

Models will appear in `/models`.

### 5. Run Streamlit app

```
streamlit run app.py
```

Then open: **[http://localhost:8501](http://localhost:8501)**

---

## Application Behavior

- Uses rule‑based diagnosis if model is missing
- Auto-generates medicine stock if file doesn't exist
- Logs all admissions to `data/admission_log.csv`

---

## Production Notes

This is a demo only.
For real deployment, you must add:

- Clinical validation
- Authentication + authorization
- Database integration
- Encryption + audit logging
- CI/CD pipeline
- Real hospital data

---

## Contributing

1. Fork repo
2. Create feature branch
3. Commit changes
4. Open Pull Request

Include change summary and testing steps.

---

## License

Created By Team - 7 @2025
