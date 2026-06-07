"""
==========================================================================
SCRIPT 06: SHAP Analysis + Model Comparison Visualizations
==========================================================================
SHAP (SHapley Additive exPlanations) explains WHY each model makes its
prediction. Output: feature importance, summary plots, and comparison
charts between ANN and XGBoost.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import json

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
sns.set_style('whitegrid')

# ==========================================================================
# 1. LOAD
# ==========================================================================
data = np.load('xgb_predictions.npz', allow_pickle=True)
feature_names = data['feature_names'].tolist()

with open('all_model_metrics.json') as f:
    all_metrics = json.load(f)

# Reload data for SHAP (we need the X matrix)
df = pd.read_csv('milano_cleaned.csv', low_memory=False)
num_features = ['SLA_hours', 'urgency_level',
                'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
                'hour_sin', 'hour_cos']
df['Management'] = df['Management'].fillna('UNK')
mgmt_dummies = pd.get_dummies(df['Management'], prefix='mgmt', dtype=int)
cat_dummies = pd.get_dummies(df['macro_category'], prefix='cat', dtype=int)
building_codes = df['ID_Building'].astype('category').cat.codes.values

X = pd.concat([
    df[num_features].astype(float),
    mgmt_dummies, cat_dummies,
    pd.Series(building_codes, name='building_code', index=df.index)
], axis=1)

from sklearn.model_selection import train_test_split
y_clf = df['is_delayed'].astype(int).values
idx = np.arange(len(df))
idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=42, stratify=y_clf)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y_clf[idx_temp])
X_test = X.iloc[idx_test]

# Load XGBoost models
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model('model_xgb_classification.json')
xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model('model_xgb_regression.json')

# ==========================================================================
# 2. SHAP for CLASSIFIER
# ==========================================================================
print("Computing SHAP values for classifier...")
explainer_clf = shap.TreeExplainer(xgb_clf)
# Sample 1500 rows for speed
sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(1500, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx]
shap_values_clf = explainer_clf.shap_values(X_sample)

# Top features by mean |SHAP|
mean_abs_shap_clf = np.abs(shap_values_clf).mean(axis=0)
shap_importance_clf = pd.DataFrame({
    'feature': feature_names,
    'shap_importance': mean_abs_shap_clf
}).sort_values('shap_importance', ascending=False)

print("\n=== Top 15 most important features for CLASSIFIER ===")
print(shap_importance_clf.head(15).to_string(index=False))

# ==========================================================================
# 3. SHAP for REGRESSOR
# ==========================================================================
print("\nComputing SHAP values for regressor...")
explainer_reg = shap.TreeExplainer(xgb_reg)
shap_values_reg = explainer_reg.shap_values(X_sample)

mean_abs_shap_reg = np.abs(shap_values_reg).mean(axis=0)
shap_importance_reg = pd.DataFrame({
    'feature': feature_names,
    'shap_importance': mean_abs_shap_reg
}).sort_values('shap_importance', ascending=False)

print("\n=== Top 15 most important features for REGRESSOR ===")
print(shap_importance_reg.head(15).to_string(index=False))

# Save importance tables
shap_importance_clf.to_csv('shap_importance_classification.csv', index=False)
shap_importance_reg.to_csv('shap_importance_regression.csv', index=False)

# ==========================================================================
# 4. FIGURE 8: SHAP Summary Plot — Classification
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle('SHAP Feature Importance — Top 15 Features', fontsize=14, fontweight='bold')

# Classification
ax = axes[0]
top15_clf = shap_importance_clf.head(15).iloc[::-1]   # reverse for bar plot
ax.barh(top15_clf['feature'], top15_clf['shap_importance'],
        color='#3F51B5', edgecolor='black')
ax.set_title('Classification Model')
ax.set_xlabel('Mean |SHAP value|')
for i, v in enumerate(top15_clf['shap_importance']):
    ax.text(v * 1.02, i, f'{v:.3f}', va='center', fontsize=8)

# Regression
ax = axes[1]
top15_reg = shap_importance_reg.head(15).iloc[::-1]
ax.barh(top15_reg['feature'], top15_reg['shap_importance'],
        color='#009688', edgecolor='black')
ax.set_title('Regression Model')
ax.set_xlabel('Mean |SHAP value|')
for i, v in enumerate(top15_reg['shap_importance']):
    ax.text(v * 1.02, i, f'{v:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('fig8_shap_importance.png', bbox_inches='tight')
plt.close()
print("✓ fig8_shap_importance.png saved")

# ==========================================================================
# 5. FIGURE 9: SHAP Beeswarm summary (classification)
# ==========================================================================
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_clf, X_sample, feature_names=feature_names,
                  max_display=15, show=False, plot_size=(10, 8))
plt.title('SHAP Summary — Classification (impact direction + magnitude)',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig9_shap_beeswarm.png', bbox_inches='tight')
plt.close()
print("✓ fig9_shap_beeswarm.png saved")

# ==========================================================================
# 6. FIGURE 10: Model Comparison — ANN vs XGBoost
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Comparison: ANN vs XGBoost', fontsize=14, fontweight='bold')

# Classification metrics comparison
ax = axes[0]
clf_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
ann_vals = [all_metrics['ANN']['classification'][m] for m in clf_metrics]
xgb_vals = [all_metrics['XGBoost']['classification'][m] for m in clf_metrics]
x = np.arange(len(clf_metrics))
w = 0.35
b1 = ax.bar(x - w/2, ann_vals, w, label='ANN', color='#FF7043', edgecolor='black')
b2 = ax.bar(x + w/2, xgb_vals, w, label='XGBoost', color='#3F51B5', edgecolor='black')
for b, v in zip(b1, ann_vals): ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=8)
for b, v in zip(b2, xgb_vals): ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(clf_metrics, rotation=20)
ax.set_ylim(0, 1); ax.set_ylabel('Score')
ax.set_title('Classification Performance')
ax.legend()

# Regression metrics comparison (normalized for visibility)
ax = axes[1]
reg_metrics = ['R²', 'Val R²']
ann_reg = [all_metrics['ANN']['regression'][m] for m in reg_metrics]
xgb_reg_v = [all_metrics['XGBoost']['regression'][m] for m in reg_metrics]
x = np.arange(len(reg_metrics))
b1 = ax.bar(x - w/2, ann_reg, w, label='ANN', color='#FF7043', edgecolor='black')
b2 = ax.bar(x + w/2, xgb_reg_v, w, label='XGBoost', color='#3F51B5', edgecolor='black')
for b, v in zip(b1, ann_reg): ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontweight='bold')
for b, v in zip(b2, xgb_reg_v): ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(reg_metrics)
ax.set_ylim(0, max(max(ann_reg), max(xgb_reg_v)) * 1.2)
ax.set_ylabel('R² Score')
ax.set_title('Regression Performance')
ax.legend()

plt.tight_layout()
plt.savefig('fig10_model_comparison.png', bbox_inches='tight')
plt.close()
print("✓ fig10_model_comparison.png saved")

# ==========================================================================
# 7. FIGURE 11: XGBoost regression predicted vs actual
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('XGBoost Regression — Predicted vs Ground Truth Delay (hours)',
             fontsize=14, fontweight='bold')

# Val
ax = axes[0]
val_r2 = all_metrics['XGBoost']['regression']['Val R²']
ax.scatter(data['y_reg_val'], data['y_pred_val_xgb'], s=15, alpha=0.45,
           color='#3F51B5', edgecolor='none')
lims = [min(data['y_reg_val'].min(), data['y_pred_val_xgb'].min()),
        max(data['y_reg_val'].max(), data['y_pred_val_xgb'].max())]
ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Delay ground truth [h]')
ax.set_ylabel('Predicted Delay [h]')
ax.set_title(f'on validation data    $R^2 = {val_r2:.3f}$')
ax.legend()

# Test
ax = axes[1]
test_r2 = all_metrics['XGBoost']['regression']['R²']
ax.scatter(data['y_reg_test'], data['y_pred_reg_xgb'], s=15, alpha=0.45,
           color='#009688', edgecolor='none')
lims = [min(data['y_reg_test'].min(), data['y_pred_reg_xgb'].min()),
        max(data['y_reg_test'].max(), data['y_pred_reg_xgb'].max())]
ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Delay ground truth [h]')
ax.set_ylabel('Predicted Delay [h]')
ax.set_title(f'on test data    $R^2 = {test_r2:.3f}$')
ax.legend()

mae = all_metrics['XGBoost']['regression']['MAE']
rmse = all_metrics['XGBoost']['regression']['RMSE']
fig.text(0.5, -0.02, f"Test R²: {test_r2:.3f}  |  MAE: {mae:.0f}h  |  RMSE: {rmse:.0f}h",
         ha='center', fontsize=11, fontweight='bold', color='#333')
plt.tight_layout()
plt.savefig('fig11_xgb_regression.png', bbox_inches='tight')
plt.close()
print("✓ fig11_xgb_regression.png saved")

print("\n" + "=" * 70)
print("ALL SHAP & COMPARISON VISUALIZATIONS COMPLETE")
print("=" * 70)
