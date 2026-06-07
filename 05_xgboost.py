"""
==========================================================================
SCRIPT 05: XGBoost Models + SHAP Analysis
==========================================================================
We train two XGBoost models (classification + regression) and compare them
against the ANN baselines. XGBoost is known to outperform neural networks
on tabular data with mixed feature types, which is exactly our setting.

SHAP (SHapley Additive exPlanations) provides game-theoretic feature
importance, telling us WHY the model makes each prediction.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             r2_score, mean_absolute_error, mean_squared_error)
import xgboost as xgb
import shap

np.random.seed(42)

# ==========================================================================
# 1. LOAD CLEANED DATA
# ==========================================================================
print("=" * 70)
print("XGBoost + SHAP Analysis")
print("=" * 70)
df = pd.read_csv('milano_cleaned.csv', low_memory=False)
print(f"Loaded: {len(df):,} records")

# ==========================================================================
# 2. FEATURE PREPARATION (XGBoost handles categoricals natively)
# ==========================================================================
num_features = ['SLA_hours', 'urgency_level',
                'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
                'hour_sin', 'hour_cos']

df['Management'] = df['Management'].fillna('UNK')
mgmt_dummies = pd.get_dummies(df['Management'], prefix='mgmt', dtype=int)
cat_dummies = pd.get_dummies(df['macro_category'], prefix='cat', dtype=int)

# Building as label-encoded integer (XGBoost can handle this natively)
building_codes = df['ID_Building'].astype('category').cat.codes.values

# Build the full feature matrix
X = pd.concat([
    df[num_features].astype(float),
    mgmt_dummies, cat_dummies,
    pd.Series(building_codes, name='building_code', index=df.index)
], axis=1)

print(f"Total features: {X.shape[1]}")
feature_names = X.columns.tolist()

y_clf = df['is_delayed'].astype(int).values
y_reg = df['delay_hours'].astype(float).values

# Train/Val/Test split (same as ANN for fair comparison)
idx = np.arange(len(df))
idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=42, stratify=y_clf)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y_clf[idx_temp])

X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
y_clf_train, y_clf_val, y_clf_test = y_clf[idx_train], y_clf[idx_val], y_clf[idx_test]
y_reg_train, y_reg_val, y_reg_test = y_reg[idx_train], y_reg[idx_val], y_reg[idx_test]

print(f"Train/Val/Test: {len(idx_train):,} / {len(idx_val):,} / {len(idx_test):,}")

# ==========================================================================
# 3. XGBoost CLASSIFICATION
# ==========================================================================
print("\n" + "=" * 70)
print("XGBoost — CLASSIFICATION")
print("=" * 70)

xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)

xgb_clf.fit(X_train, y_clf_train,
            eval_set=[(X_train, y_clf_train), (X_val, y_clf_val)],
            verbose=False)

print(f"Best iteration: {xgb_clf.best_iteration}")

# Evaluation
y_pred_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
y_pred_clf_xgb = (y_pred_proba_xgb >= 0.5).astype(int)

clf_metrics_xgb = {
    'Accuracy' : float(accuracy_score(y_clf_test, y_pred_clf_xgb)),
    'Precision': float(precision_score(y_clf_test, y_pred_clf_xgb)),
    'Recall'   : float(recall_score(y_clf_test, y_pred_clf_xgb)),
    'F1-score' : float(f1_score(y_clf_test, y_pred_clf_xgb)),
    'ROC-AUC'  : float(roc_auc_score(y_clf_test, y_pred_proba_xgb)),
}

print("\n--- XGBoost Classification Test Performance ---")
for k, v in clf_metrics_xgb.items():
    print(f"  {k:10s}: {v:.4f}")

# ==========================================================================
# 4. XGBoost REGRESSION (with log transform target)
# ==========================================================================
print("\n" + "=" * 70)
print("XGBoost — REGRESSION")
print("=" * 70)

# Log-transform the target to handle heavy tail
shift = abs(y_reg_train.min()) + 1
y_reg_train_t = np.log1p(y_reg_train + shift)
y_reg_val_t   = np.log1p(y_reg_val + shift)
y_reg_test_t  = np.log1p(y_reg_test + shift)

xgb_reg = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='reg:squarederror',
    early_stopping_rounds=25,
    random_state=42,
    n_jobs=-1
)

xgb_reg.fit(X_train, y_reg_train_t,
            eval_set=[(X_train, y_reg_train_t), (X_val, y_reg_val_t)],
            verbose=False)

print(f"Best iteration: {xgb_reg.best_iteration}")

y_pred_reg_t_xgb = xgb_reg.predict(X_test)
y_pred_reg_xgb = np.expm1(y_pred_reg_t_xgb) - shift

y_pred_val_t_xgb = xgb_reg.predict(X_val)
y_pred_val_xgb = np.expm1(y_pred_val_t_xgb) - shift

reg_metrics_xgb = {
    'R²'    : float(r2_score(y_reg_test, y_pred_reg_xgb)),
    'MAE'   : float(mean_absolute_error(y_reg_test, y_pred_reg_xgb)),
    'RMSE'  : float(np.sqrt(mean_squared_error(y_reg_test, y_pred_reg_xgb))),
    'Val R²': float(r2_score(y_reg_val, y_pred_val_xgb)),
}

print("\n--- XGBoost Regression Test Performance ---")
for k, v in reg_metrics_xgb.items():
    print(f"  {k:6s}: {v:.4f}")

# ==========================================================================
# 5. COMPARISON WITH ANN
# ==========================================================================
print("\n" + "=" * 70)
print("COMPARISON: ANN vs XGBoost")
print("=" * 70)

with open('model_metrics.json') as f:
    ann_metrics = json.load(f)

print("\n--- Classification ---")
print(f"{'Metric':12s} {'ANN':>10s} {'XGBoost':>10s} {'Δ':>10s}")
for k in clf_metrics_xgb.keys():
    ann_v = ann_metrics['classification'][k]
    xgb_v = clf_metrics_xgb[k]
    delta = xgb_v - ann_v
    sign = '+' if delta >= 0 else ''
    print(f"{k:12s} {ann_v:>10.4f} {xgb_v:>10.4f} {sign}{delta:>9.4f}")

print("\n--- Regression ---")
print(f"{'Metric':12s} {'ANN':>10s} {'XGBoost':>10s} {'Δ':>10s}")
for k in ['R²', 'MAE', 'RMSE', 'Val R²']:
    ann_v = ann_metrics['regression'][k]
    xgb_v = reg_metrics_xgb[k]
    delta = xgb_v - ann_v
    sign = '+' if delta >= 0 else ''
    print(f"{k:12s} {ann_v:>10.4f} {xgb_v:>10.4f} {sign}{delta:>9.4f}")

# ==========================================================================
# 6. SAVE
# ==========================================================================
all_metrics = {
    'ANN':     ann_metrics,
    'XGBoost': {
        'classification': clf_metrics_xgb,
        'regression':     reg_metrics_xgb,
    }
}
with open('all_model_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

xgb_clf.save_model('model_xgb_classification.json')
xgb_reg.save_model('model_xgb_regression.json')

np.savez('xgb_predictions.npz',
         y_clf_test=y_clf_test, y_pred_proba_xgb=y_pred_proba_xgb, y_pred_clf_xgb=y_pred_clf_xgb,
         y_reg_test=y_reg_test, y_pred_reg_xgb=y_pred_reg_xgb,
         y_reg_val=y_reg_val, y_pred_val_xgb=y_pred_val_xgb,
         feature_names=np.array(feature_names))

print("\n✓ Saved: XGBoost models + metrics + predictions")
