"""
==========================================================================
SCRIPT 04: Visualize Model Performance
==========================================================================
Mimics PDF page 15: scatter plots of predicted vs ground truth + metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             r2_score, mean_absolute_error)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
sns.set_style('whitegrid')

# Load predictions
data = np.load('predictions.npz')
with open('model_metrics.json') as f:
    metrics = json.load(f)

# ==========================================================================
# FIG 4: Training Curves
# ==========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle('Model Training History', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(data['hist_clf_loss'], label='Train', color='#2196F3', linewidth=2)
ax.plot(data['hist_clf_val_loss'], label='Validation', color='#F44336', linewidth=2)
ax.set_title('Classification — Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Binary Cross-Entropy')
ax.legend()

ax = axes[1]
ax.plot(data['hist_clf_auc'], label='Train', color='#2196F3', linewidth=2)
ax.plot(data['hist_clf_val_auc'], label='Validation', color='#F44336', linewidth=2)
ax.set_title('Classification — ROC-AUC')
ax.set_xlabel('Epoch')
ax.set_ylabel('AUC')
ax.legend()

ax = axes[2]
ax.plot(data['hist_reg_loss'], label='Train', color='#2196F3', linewidth=2)
ax.plot(data['hist_reg_val_loss'], label='Validation', color='#F44336', linewidth=2)
ax.set_title('Regression — Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE (normalized)')
ax.legend()

plt.tight_layout()
plt.savefig('fig4_training_history.png', bbox_inches='tight')
plt.close()
print("✓ fig4_training_history.png")

# ==========================================================================
# FIG 5: Classification Performance
# ==========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Classification Model — Test Performance', fontsize=14, fontweight='bold')

# (1) Confusion matrix
ax = axes[0]
cm = confusion_matrix(data['y_clf_test'], data['y_pred_clf'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['On Time', 'Delayed'], yticklabels=['On Time', 'Delayed'],
            cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# (2) ROC curve
ax = axes[1]
fpr, tpr, _ = roc_curve(data['y_clf_test'], data['y_pred_proba'])
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color='#3F51B5', linewidth=2.5, label=f'ANN (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.fill_between(fpr, tpr, alpha=0.15, color='#3F51B5')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')

# (3) Probability distribution by true class
ax = axes[2]
ax.hist(data['y_pred_proba'][data['y_clf_test']==0], bins=30, alpha=0.6,
        color='#4CAF50', label='Actual: On Time', edgecolor='black')
ax.hist(data['y_pred_proba'][data['y_clf_test']==1], bins=30, alpha=0.6,
        color='#F44336', label='Actual: Delayed', edgecolor='black')
ax.axvline(0.5, color='black', linestyle='--', label='Decision threshold')
ax.set_xlabel('Predicted Probability of Delay')
ax.set_ylabel('Count')
ax.set_title('Predicted Probability by True Class')
ax.legend()

# Show metrics as text on figure
clf = metrics['classification']
metric_text = (f"Accuracy: {clf['Accuracy']:.3f}  |  "
               f"Precision: {clf['Precision']:.3f}  |  "
               f"Recall: {clf['Recall']:.3f}  |  "
               f"F1: {clf['F1-score']:.3f}  |  "
               f"AUC: {clf['ROC-AUC']:.3f}")
fig.text(0.5, -0.02, metric_text, ha='center', fontsize=11,
         fontweight='bold', color='#333')

plt.tight_layout()
plt.savefig('fig5_classification.png', bbox_inches='tight')
plt.close()
print("✓ fig5_classification.png")

# ==========================================================================
# FIG 6: Regression Performance — mirrors PDF page 15
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Regression Model — Predicted vs Ground Truth Delay (hours)',
             fontsize=14, fontweight='bold')

# Validation
ax = axes[0]
val_r2 = metrics['regression']['Val R²']
ax.scatter(data['y_reg_val'], data['y_pred_val'], s=15, alpha=0.45,
           color='#3F51B5', edgecolor='none')
lims = [min(data['y_reg_val'].min(), data['y_pred_val'].min()),
        max(data['y_reg_val'].max(), data['y_pred_val'].max())]
ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Delay ground truth [h]')
ax.set_ylabel('Predicted Delay [h]')
ax.set_title(f'on validation data    $R^2 = {val_r2:.3f}$')
ax.legend()

# Test
ax = axes[1]
test_r2 = metrics['regression']['R²']
ax.scatter(data['y_reg_test'], data['y_pred_reg'], s=15, alpha=0.45,
           color='#009688', edgecolor='none')
lims = [min(data['y_reg_test'].min(), data['y_pred_reg'].min()),
        max(data['y_reg_test'].max(), data['y_pred_reg'].max())]
ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Delay ground truth [h]')
ax.set_ylabel('Predicted Delay [h]')
ax.set_title(f'on test data    $R^2 = {test_r2:.3f}$')
ax.legend()

reg_text = (f"Test R²: {test_r2:.3f}  |  "
            f"MAE: {metrics['regression']['MAE']:.0f}h  |  "
            f"RMSE: {metrics['regression']['RMSE']:.0f}h")
fig.text(0.5, -0.02, reg_text, ha='center', fontsize=11,
         fontweight='bold', color='#333')

plt.tight_layout()
plt.savefig('fig6_regression.png', bbox_inches='tight')
plt.close()
print("✓ fig6_regression.png")

# ==========================================================================
# FIG 7: Residual analysis for regression
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Regression Residual Analysis', fontsize=14, fontweight='bold')

residuals = data['y_reg_test'] - data['y_pred_reg']

# (1) Residual vs Predicted
ax = axes[0]
ax.scatter(data['y_pred_reg'], residuals, s=15, alpha=0.4,
           color='#FF5722', edgecolor='none')
ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
ax.set_xlabel('Predicted Delay [h]')
ax.set_ylabel('Residual (Actual - Predicted) [h]')
ax.set_title('Residuals vs Predictions')

# (2) Residual histogram
ax = axes[1]
res_clip = np.clip(residuals, -1000, 1000)
ax.hist(res_clip, bins=60, color='#FF5722', edgecolor='black', alpha=0.85)
ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax.set_xlabel('Residual [h] (clipped at ±1000h)')
ax.set_ylabel('Count')
ax.set_title(f'Residual Distribution (mean = {residuals.mean():.0f}h, std = {residuals.std():.0f}h)')

plt.tight_layout()
plt.savefig('fig7_residuals.png', bbox_inches='tight')
plt.close()
print("✓ fig7_residuals.png")

print("\n" + "=" * 70)
print("ALL VISUALIZATIONS COMPLETE")
print("=" * 70)
