import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

# Ensure reproducible behaviour
RANDOM_STATE = 42

def calculate_severity_score_for_training(df: pd.DataFrame) -> pd.Series:
    severity_score = pd.Series(0, index=df.index)
    severity_score += ((df['blood_pressure_sys'] > 180) | (df['blood_pressure_sys'] < 90)).astype(int) * 2
    severity_score += ((df['blood_pressure_dia'] > 120) | (df['blood_pressure_dia'] < 60)).astype(int) * 2
    severity_score += ((df['heart_rate'] > 120) | (df['heart_rate'] < 50)).astype(int) * 2
    severity_score += (df['temperature'] > 39.0).astype(int) * 2
    severity_score += (df['symptom_cough'] | df['symptom_fever'] | df['symptom_breathless']).astype(int)
    severity_score += (df['symptom_cough'] & df['symptom_fever']).astype(int)
    severity_score += (df['symptom_cough'] & df['symptom_breathless']).astype(int)
    severity_score += (df['symptom_fever'] & df['symptom_breathless']).astype(int) * 2
    severity_score += (df['lab_wbc'] > 15.0).astype(int) * 3
    severity_score += ((df['lab_wbc'] > 11.0) & (df['lab_wbc'] <= 15.0)).astype(int) * 2
    severity_score += (df['lab_crp'] > 50).astype(int) * 3
    severity_score += ((df['lab_crp'] > 20) & (df['lab_crp'] <= 50)).astype(int) * 2
    severity_score += ((df['lab_crp'] > 10) & (df['lab_crp'] <= 20)).astype(int)
    # comorbidities
    severity_score += df['comorbidity_diabetes'].astype(int)
    severity_score += df['comorbidity_hypertension'].astype(int)
    severity_score += (df['age'] > 65).astype(int)
    return severity_score

def prepare_features(df: pd.DataFrame):
    """
    Prepare feature matrix and engineered features for model training.
    Returns X (DataFrame of features) and y dict for various targets.
    """
    df = df.copy()
    # Ensure endocrine flags exist (generated dataset includes engineered flags)
    engineered = ['is_respiratory','is_infection','is_cardiac','is_metabolic','is_neuro','is_gi','age_over_65']
    for col in engineered:
        if col not in df.columns:
            df[col] = 0

    df['severity_score'] = calculate_severity_score_for_training(df)
    base_features = [
        'age','bmi','blood_pressure_sys','blood_pressure_dia','heart_rate','temperature',
        'symptom_cough','symptom_fever','symptom_breathless','comorbidity_diabetes','comorbidity_hypertension',
        'lab_wbc','lab_crp','severity_score'
    ]
    # Add engineered flags
    base_features += engineered

    X = df[base_features].copy()
    # include diagnosis_code in models where needed by caller

    y = {
        'diagnosis': df['diagnosis_code'].astype(str),
        'inpatient': df['inpatient'].astype(int),
        'ward': df['ward_type'].astype(str),
        'stay': df['stay_days'].astype(int),
        'medicines': df['medicines'].fillna('').astype(str)  # pipe-separated medicine lists
    }
    return X, y, df

def train_diagnosis(X, y_diag, out_dir):
    print("Training diagnosis model (multi-class)...")
    # All inputs numeric -> no encoder needed
    # But we standardize numeric features
    numeric_cols = X.columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols)
    ])
    clf = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced', max_depth=18, min_samples_leaf=3))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y_diag, test_size=0.2, stratify=y_diag, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Diagnosis Model Accuracy - Train: {train_acc:.3f}, Test: {test_acc:.3f}")
    # Save
    joblib.dump(clf, os.path.join(out_dir, 'diagnosis_model.pkl'))
    print("Saved diagnosis_model.pkl")
    # report
    y_pred = clf.predict(X_test)
    print("Diagnosis classification report (test set):")
    print(classification_report(y_test, y_pred, zero_division=0))
    return clf

def train_inpatient(X_with_diag, y_inp, out_dir):
    print("Training inpatient model (binary)...")
    # X_with_diag must include diagnosis_code column
    cat_cols = ['diagnosis_code'] if 'diagnosis_code' in X_with_diag.columns else []
    num_cols = [c for c in X_with_diag.columns if c not in cat_cols]
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])
    clf = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced', max_depth=12, min_samples_leaf=3))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X_with_diag, y_inp, test_size=0.2, stratify=y_inp, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    print(f"Inpatient Model Accuracy - Train: {clf.score(X_train, y_train):.3f}, Test: {clf.score(X_test, y_test):.3f}")
    joblib.dump(clf, os.path.join(out_dir, 'inpatient_model.pkl'))
    print("Saved inpatient_model.pkl")
    return clf

def train_ward_and_stay(X_with_diag, y_ward, y_stay, out_dir):
    print("Training ward model (multi-class) and stay model (regression)")
    cat_cols = ['diagnosis_code'] if 'diagnosis_code' in X_with_diag.columns else []
    num_cols = [c for c in X_with_diag.columns if c not in cat_cols]
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    # Ward classifier
    ward_clf = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, max_depth=12, min_samples_leaf=2))
    ])
    # Only train ward/stay on inpatients with valid ward
    mask = (y_ward != 'None') & (y_ward.notna())
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_with_diag[mask], y_ward[mask], test_size=0.2, random_state=RANDOM_STATE, stratify=y_ward[mask])
    ward_clf.fit(X_train_w, y_train_w)
    print(f"Ward Model Accuracy - Train: {ward_clf.score(X_train_w, y_train_w):.3f}, Test: {ward_clf.score(X_test_w, y_test_w):.3f}")
    joblib.dump(ward_clf, os.path.join(out_dir, 'ward_model.pkl'))
    print("Saved ward_model.pkl")

    # Stay regressor
    stay_reg = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, max_depth=12, min_samples_leaf=2))
    ])
    # Use same mask but ensure stay values are numeric
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_with_diag[mask], y_stay[mask], test_size=0.2, random_state=RANDOM_STATE)
    stay_reg.fit(X_train_s, y_train_s)
    train_mse = mean_squared_error(y_train_s, stay_reg.predict(X_train_s))
    test_mse = mean_squared_error(y_test_s, stay_reg.predict(X_test_s))
    print(f"Stay Model MSE - Train: {train_mse:.1f}, Test: {test_mse:.1f}")
    joblib.dump(stay_reg, os.path.join(out_dir, 'stay_model.pkl'))
    print("Saved stay_model.pkl")
    return ward_clf, stay_reg

def train_medicine_recommender(X_with_diag, y_meds_series, out_dir):
    """
    Train a multi-label medicine recommender.
    y_meds_series contains pipe-separated medicine strings.
    We'll convert to a multilabel binary matrix using MultiLabelBinarizer.
    """
    print("Training medicine recommender (multi-label)...")
    # prepare y: split pipes into list
    y_lists = y_meds_series.fillna('').apply(lambda s: [m.strip() for m in s.split("|") if m.strip() != ""]).tolist()
    mlb = MultiLabelBinarizer(sparse_output=False)
    Y = mlb.fit_transform(y_lists)
    if Y.shape[1] == 0:
        print("No medicine labels found; skipping medicine model training.")
        return None

    # Preprocessor similar to inpatient, include diagnosis as categorical
    cat_cols = ['diagnosis_code'] if 'diagnosis_code' in X_with_diag.columns else []
    num_cols = [c for c in X_with_diag.columns if c not in cat_cols]
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    # We'll use OneVsRestClassifier around RandomForest
    base = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', max_depth=10)
    clf = Pipeline([
        ('pre', preprocessor),
        ('clf', OneVsRestClassifier(base, n_jobs=-1))
    ])

    X_train, X_test, Y_train, Y_test = train_test_split(X_with_diag, Y, test_size=0.2, random_state=RANDOM_STATE)
    clf.fit(X_train, Y_train)
    # Evaluate: per-label accuracy (simple)
    Y_pred = clf.predict(X_test)
    acc = (Y_pred == Y_test).mean()
    print(f"Medicine recommender label-wise accuracy (avg): {acc:.3f}")
    # Save model + mlb
    joblib.dump({'pipeline': clf, 'mlb': mlb}, os.path.join(out_dir, 'medicine_model.pkl'))
    print("Saved medicine_model.pkl")
    return clf, mlb

def build_and_save_models(data_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"Training on {len(df)} samples from {data_path}")

    X_base, y_dict, df_full = prepare_features(df)

    # Train diagnosis model
    diag_clf = train_diagnosis(X_base, y_dict['diagnosis'], out_dir)

    # For models requiring diagnosis as input, add diagnosis_code into X
    X_with_diag = X_base.copy()
    X_with_diag['diagnosis_code'] = y_dict['diagnosis']

    # Train inpatient
    inp_clf = train_inpatient(X_with_diag, y_dict['inpatient'], out_dir)

    # Train ward & stay models on inpatients
    ward_clf, stay_reg = train_ward_and_stay(X_with_diag, y_dict['ward'], y_dict['stay'], out_dir)

    # Train medicine recommender
    med_model = train_medicine_recommender(X_with_diag, y_dict['medicines'], out_dir)

    print("All requested models trained and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/patients_sample.csv', help='path to training csv')
    parser.add_argument('--out_dir', type=str, default='models', help='output directory for models')
    args = parser.parse_args()
    build_and_save_models(args.data, args.out_dir)
