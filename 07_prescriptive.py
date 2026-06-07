"""
==========================================================================
SCRIPT 07: Prescriptive Analytics
==========================================================================
Managerial decision-support module built on top of the XGBoost models:

  Resource Allocation Optimization
     - Compute a Risk Priority Score (RPS) for each ticket
     - Simulate the gain from prioritizing the top-N at-risk tickets
     - RPS = P(delay) x Predicted_Delay_Hours x Urgency_Multiplier
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import json

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
sns.set_style('whitegrid')

# ==========================================================================
# 1. LOAD DATA AND MODELS
# ==========================================================================
print("=" * 70); print("PRESCRIPTIVE ANALYTICS"); print("=" * 70)

df = pd.read_csv('milano_cleaned.csv', low_memory=False)
num_features = ['SLA_hours', 'urgency_level',
                'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
                'hour_sin', 'hour_cos']
df['Management'] = df['Management'].fillna('UNK')
mgmt_dummies = pd.get_dummies(df['Management'], prefix='mgmt', dtype=int)
cat_dummies  = pd.get_dummies(df['macro_category'], prefix='cat', dtype=int)
building_codes = df['ID_Building'].astype('category').cat.codes.values

X = pd.concat([
    df[num_features].astype(float),
    mgmt_dummies, cat_dummies,
    pd.Series(building_codes, name='building_code', index=df.index)
], axis=1)

xgb_clf = xgb.XGBClassifier(); xgb_clf.load_model('model_xgb_classification.json')
xgb_reg = xgb.XGBRegressor();  xgb_reg.load_model('model_xgb_regression.json')

df['pred_proba']       = xgb_clf.predict_proba(X)[:, 1]
shift_val              = abs(df['delay_hours'].min()) + 1
df['pred_delay_log']   = xgb_reg.predict(X)
df['pred_delay_hours'] = (np.expm1(df['pred_delay_log']) - shift_val).clip(lower=0)
print(f"Predictions generated for {len(df):,} tickets")

# ==========================================================================
# 2. RESOURCE ALLOCATION OPTIMIZATION
# ==========================================================================
print("\n" + "=" * 70)
print("RESOURCE ALLOCATION — Risk Priority Score (RPS)")
print("=" * 70)
print("RPS = P(delay) x Predicted_Delay_Hours x Urgency_Multiplier")

urgency_mult = {0: 1.0, 1: 1.5, 2: 2.0}
df['urgency_mult'] = df['urgency_level'].map(urgency_mult)
df['RPS'] = df['pred_proba'] * df['pred_delay_hours'] * df['urgency_mult']

print(f"\nRPS statistics:")
print(f"  Median : {df['RPS'].median():.1f}")
print(f"  Top 20% threshold: {df['RPS'].quantile(0.80):.1f}")
print(f"  Top 10% threshold: {df['RPS'].quantile(0.90):.1f}")

# Simulate: for each targeting fraction, halve the delay of targeted tickets
total_delay_baseline = df['delay_hours'].clip(lower=0).sum()
scenarios = []
for top_pct in [5, 10, 15, 20, 25, 30, 40, 50]:
    threshold = df['RPS'].quantile(1 - top_pct / 100)
    targeted  = df['RPS'] >= threshold
    new_delays = df['delay_hours'].clip(lower=0).copy()
    new_delays[targeted] *= 0.5   # conservative assumption: 50% delay reduction
    new_total = new_delays.sum()
    reduction_pct = (total_delay_baseline - new_total) / total_delay_baseline * 100
    scenarios.append({
        'top_pct':          top_pct,
        'targeted_count':   int(targeted.sum()),
        'new_total_delay_h': new_total,
        'reduction_pct':    reduction_pct
    })

resource_scenarios = pd.DataFrame(scenarios)
print("\nScenario results (assumption: 50% delay reduction on targeted tickets):")
print(resource_scenarios.to_string(index=False))
resource_scenarios.to_csv('prescriptive_resource_scenarios.csv', index=False)

# ==========================================================================
# 3. VISUALIZATION — Figure 12
# ==========================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(resource_scenarios['top_pct'], resource_scenarios['reduction_pct'],
        marker='o', linewidth=2.5, markersize=10, color='#3F51B5')
ax.fill_between(resource_scenarios['top_pct'], 0,
                resource_scenarios['reduction_pct'], alpha=0.2, color='#3F51B5')
for _, r in resource_scenarios.iterrows():
    ax.annotate(f'{r["reduction_pct"]:.1f}%',
                (r['top_pct'], r['reduction_pct']),
                textcoords='offset points', xytext=(0, 12),
                ha='center', fontweight='bold', fontsize=9)
ax.set_xlabel('Top % of tickets prioritized by RPS')
ax.set_ylabel('Total Delay Reduction (%)')
ax.set_title('Resource Allocation Impact:\n'
             'Reduction in Total Delay vs. Fraction of Tickets Prioritized',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig12_resource_allocation.png', bbox_inches='tight')
plt.close()
print("\n✓ fig12_resource_allocation.png saved")

# ==========================================================================
# 4. SAVE SUMMARY
# ==========================================================================
prescriptive_summary = {
    'resource_allocation': resource_scenarios.to_dict('records'),
}
with open('prescriptive_summary.json', 'w') as f:
    json.dump(prescriptive_summary, f, indent=2)
print("✓ prescriptive_summary.json saved")
print("\n" + "=" * 70)
print("PRESCRIPTIVE ANALYTICS COMPLETE")
print("=" * 70)
