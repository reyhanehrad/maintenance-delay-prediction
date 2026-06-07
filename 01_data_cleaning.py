"""
==========================================================================
PROJECT: Predicting Maintenance Ticket Delays - POLIMI Milano Campus
SCRIPT 01: Data Cleaning
==========================================================================
Inspired by the professor's approach (PDF pages 7-8): step-by-step cleaning
with logging of how many records survive each step.
"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
print("=" * 70)
print("STEP 0 — LOAD RAW DATA")
print("=" * 70)
df = pd.read_excel('Group_14.xlsx', sheet_name='2023-12-21_Statistiche_InfoCAD_')
print(f"Initial rows : {len(df):,}")
print(f"Initial cols : {df.shape[1]}")

records_log = [("Start (raw data)", len(df))]

# ------------------------------------------------------------------
# 2. RESTRICT TO MILANO CAMPUS
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 1 — RESTRICT TO MILANO CAMPUS")
print("=" * 70)
print("Decision: keep Milano + treat 22 NaN-city buildings as Milano")
print("Justification:")
print("  - Building IDs of NaN-city group fall within Milano range")
print("  - BuildingGroup 138 is shared between NaN-city and Milano")
print("  - Sequential IDs (138-167) right after Galbiate (137) suggest Milano")

before = len(df)
df = df[(df['City'] == 'Milano') | (df['City'].isna())].copy()
df['City'] = 'Milano'  # assign all to Milano

after = len(df)
print(f"\nRows: {before:,} -> {after:,} (kept {after/before*100:.1f}%)")
print(f"Unique buildings: {df['ID_Building'].nunique()}")
records_log.append(("After Milano filter", after))

# ------------------------------------------------------------------
# 3. ESSENTIAL VALIDITY FILTERS
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 — ESSENTIAL VALIDITY FILTERS")
print("=" * 70)

# State must be Closed (already true for all records)
mask = df['State'] == 'Closed'
df = df[mask].copy()
print(f"After State=Closed     : {len(df):,}")
records_log.append(("State = Closed", len(df)))

# All key dates must exist
date_cols = ['Requested date', 'Opening date', 'Closing Date', 'Expiration']
mask = df[date_cols].notna().all(axis=1)
df = df[mask].copy()
print(f"After all dates present: {len(df):,}")
records_log.append(("Key dates present", len(df)))

# Closing must be after Opening (sanity)
mask = df['Closing Date'] > df['Opening date']
df = df[mask].copy()
print(f"After Closing > Opening: {len(df):,}")
records_log.append(("Closing > Opening", len(df)))

# ------------------------------------------------------------------
# 4. BUILD TARGET VARIABLES
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 — BUILD TARGETS")
print("=" * 70)

# Regression target: delay in hours (positive = late, negative = early)
df['delay_hours'] = -df['Delta Close to Expiration (h)']

# Classification target: existing 'Expired' column
df['is_delayed'] = df['Expired'].astype(int)

print(f"Regression target stats (delay_hours):")
print(df['delay_hours'].describe().round(2).to_string())
print(f"\nClassification balance (is_delayed):")
print(df['is_delayed'].value_counts(normalize=True).round(3).to_string())

# ------------------------------------------------------------------
# 5. OUTLIER REMOVAL (professor's step 2 approach)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 — OUTLIER REMOVAL")
print("=" * 70)

# Remove tickets resolved in less than 5 minutes (likely test/error tickets)
mask = df['Delta Opening to Closure (h)'] >= (5/60)
df = df[mask].copy()
print(f"After resolution >= 5 min : {len(df):,}")
records_log.append(("Resolution time >= 5 min", len(df)))

# Remove extreme delays (above 99.5 percentile, likely abandoned tickets)
delay_cap = df['delay_hours'].quantile(0.995)
mask = df['delay_hours'] <= delay_cap
df = df[mask].copy()
print(f"After delay <= 99.5%-ile ({delay_cap:.0f}h) : {len(df):,}")
records_log.append(("Delay outliers removed", len(df)))

# ------------------------------------------------------------------
# 6. FEATURE ENGINEERING
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 — FEATURE ENGINEERING")
print("=" * 70)

# Convert Expiration Time to hours
expiration_map = {
    '1min': 1/60, '5min': 5/60, '15min': 15/60, '30min': 0.5,
    '45min': 0.75, '60min': 1, '2h': 2, '3h': 3, '5h': 5,
    '12h': 12, '24h': 24, '120min': 2, '10 days': 240
}
df['SLA_hours'] = df['Expiration Time'].map(expiration_map)
print(f"SLA_hours non-null: {df['SLA_hours'].notna().sum():,} / {len(df):,}")

# Encode urgency as ordinal
urgency_map = {'No emergency': 0, 'Urgency': 1, 'Emergency': 2}
df['urgency_level'] = df['Urgency'].map(urgency_map)

# Calendar features from Opening date
df['opening_month']    = df['Opening date'].dt.month
df['opening_dayofweek'] = df['Opening date'].dt.dayofweek
df['opening_hour']     = df['Opening date'].dt.hour

# Cyclical encoding (sin/cos so December and January are close)
df['month_sin'] = np.sin(2 * np.pi * df['opening_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['opening_month'] / 12)
df['dow_sin']   = np.sin(2 * np.pi * df['opening_dayofweek'] / 7)
df['dow_cos']   = np.cos(2 * np.pi * df['opening_dayofweek'] / 7)
df['hour_sin']  = np.sin(2 * np.pi * df['opening_hour'] / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['opening_hour'] / 24)

# Extract macro-category from O.I. category code (e.g., "ms.01-..." or "AV-AUDIO-VIDEO" -> "ms"/"av")
df['macro_category'] = (
    df['O.I. category for Tickets']
    .str.split('.').str[0]
    .str.split('-').str[0]
    .str.lower()
    .str.strip()
)
print(f"Macro-categories ({df['macro_category'].nunique()}): {sorted(df['macro_category'].dropna().unique().tolist())}")

print(f"\nFinal cleaned dataset: {len(df):,} rows, {df.shape[1]} cols")

# ------------------------------------------------------------------
# 7. DROP FINAL NaN ON CRITICAL FEATURES
# ------------------------------------------------------------------
critical_features = ['SLA_hours', 'urgency_level', 'macro_category', 'Management']
before = len(df)
df = df.dropna(subset=critical_features).copy()
print(f"\nAfter dropping NaN on critical features: {before:,} -> {len(df):,}")
records_log.append(("Critical features non-null", len(df)))

# ------------------------------------------------------------------
# 8. SAVE CLEANED DATA + LOG
# ------------------------------------------------------------------
df.to_csv('milano_cleaned.csv', index=False)
print(f"\n✓ Saved cleaned data to milano_cleaned.csv ({len(df):,} rows)")

log_df = pd.DataFrame(records_log, columns=['Step', 'Records'])
log_df['% remaining'] = (log_df['Records'] / log_df.iloc[0]['Records'] * 100).round(1)
log_df.to_csv('cleaning_log.csv', index=False)
print(f"\n=== CLEANING LOG ===")
print(log_df.to_string(index=False))
