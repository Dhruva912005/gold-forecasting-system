import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from utils import prepare_features, extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gold_data.db")
MODEL_FULL_PATH = os.path.join(BASE_DIR, "model.pkl")
MODEL_RECENT_PATH = os.path.join(BASE_DIR, "model_recent.pkl")

def train_and_save_models():
    print("========================================")
    print("   TRAINING GOLD PREDICTION MODELS")
    print("========================================")

    # SECURE CONNECTION TO SQLITE
    print("[1/5] Loading data from SQLite TSDB...")
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM gold_prices ORDER BY Date ASC", conn)
        conn.close()
    except Exception as e:
        print(f"Error reading SQLite database: {e}")
        return

    print(f"[2/5] Engineering technical & sentiment features against {len(df)} rows...")
    df = prepare_features(df)
    
    X, y = extract_features(df)
    
    # Strict Time-based Train-Test Splitting globally
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print("[3/5] Compiling and Training Full Historical Sequence Model...")
    model_full = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_full.fit(X_train, y_train)

    # Verify Baseline Target
    y_pred_full = model_full.predict(X_test)
    print(f"   -> Full Model Performance // MAE: {mean_absolute_error(y_test, y_pred_full):.5f} | R2: {r2_score(y_test, y_pred_full):.3f}")

    joblib.dump(model_full, MODEL_FULL_PATH)

    print("[4/5] Extracting Recent Regime Array & Training Sequential Model...")
    recent_split_idx = len(df) - 365
    if recent_split_idx < 0:
        recent_split_idx = 0
        
    X_recent = X.iloc[recent_split_idx:]
    y_recent = y.iloc[recent_split_idx:]
    
    # 80/20 of the latest 365
    rec_test_split = int(len(X_recent) * 0.8)
    X_recent_train = X_recent.iloc[:rec_test_split]
    y_recent_train = y_recent.iloc[:rec_test_split]

    model_recent = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_recent.fit(X_recent_train, y_recent_train)
    
    joblib.dump(model_recent, MODEL_RECENT_PATH)
    print("[5/5] Storage Complete. Regressors synced effectively to local env.")

if __name__ == "__main__":
    train_and_save_models()