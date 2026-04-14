import sqlite3
import pandas as pd
from datetime import datetime
from utils import prepare_features

# 1. DATABASE CHECK
conn = sqlite3.connect("gold_data.db")
df = pd.read_sql("SELECT * FROM gold_prices", conn)
conn.close()

df['Date'] = pd.to_datetime(df['Date'])
last_date = df['Date'].max()
today = pd.Timestamp.today().normalize()

print("========== PIPELINE SANITY CHECK ==========")
print(f"[1] Last Date in DB: {last_date.date()} | Today is: {today.date()}")
if last_date.date() == today.date():
    print("    ✅ Data Update Success: Reached Today")
else:
    print("    ❌ Data Update Failed: Missing Today")

print(f"[2] Data Integrity:")
print(f"    Missing Gold_Price values: {df['Gold_Price'].isnull().sum()}")
articles_count = df[["Article1", "Article2", "Article3", "Article4", "Article5"]].notnull().sum().sum()
print(f"    Total News Articles populated: {articles_count} out of {len(df)*5}")
print(f"    Duplicates: {df.duplicated(subset=['Date']).sum()}")
print(f"    Is chronologically Sorted: {df['Date'].is_monotonic_increasing}")

print(f"[3] Sliding Window Check:")
print(f"    Total Rows in DB: {len(df)}")
if len(df) <= 2500:
    print("    ✅ Sliding Window Success (<= 2500)")
else:
    print("    ❌ Sliding Window Failed")

print(f"[4] Feature Engine & Model Input State:")
try:
    df_prepared = prepare_features(df)
    last_prep_date = df_prepared['Date'].max()
    print(f"    ✅ Prepared Features Generated Successfully")
    print(f"    Prepared Features Last Date: {last_prep_date.date()}")
    print(f"    Next valid prediction sequence strictly begins AFTER {last_prep_date.date()}")
except Exception as e:
    print(f"    ❌ Feature Compilation Failure: {e}")

print("===========================================")
