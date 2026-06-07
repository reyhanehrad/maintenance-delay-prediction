"""
==========================================================================
SCRIPT 03 v2: ANN Models with STRONGER REGULARIZATION
==========================================================================
Goal: bring Train/Val curves closer together (less overfitting), even at
the cost of slightly lower peak performance.

Key changes vs v1:
- Dropout 0.50 -> 0.20 (was 0.30 -> 0.10)
- L2 regularization 1e-3 (was 1e-5)
- Embedding dim 4 (was 8)
- Learning rate 5e-4 (was 1e-3)
- Smaller architecture: 256->128->64->32->16 (was 512->...->16)
- More aggressive LR reduction
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score,
                             r2_score, mean_absolute_error, mean_squared_error)
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
tf.random.set_seed(42); np.random.seed(42)

# ==========================================================================
# 1. LOAD CLEANED DATA
# ==========================================================================
print("=" * 70)
print("ANN v2 — Stronger Regularization")
print("=" * 70)
df = pd.read_csv('milano_cleaned.csv', low_memory=False)
print(f"Loaded: {len(df):,} records")

# ==========================================================================
# 2. PREPARE FEATURES (same as v1)
# ==========================================================================
num_features = ['SLA_hours', 'urgency_level',
                'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
                'hour_sin', 'hour_cos']
df['Management'] = df['Management'].fillna('UNK')
mgmt_dummies = pd.get_dummies(df['Management'], prefix='mgmt', dtype=float)
cat_dummies  = pd.get_dummies(df['macro_category'], prefix='cat', dtype=float)
building_ids = df['ID_Building'].astype('category')
building_codes = building_ids.cat.codes.values
n_buildings = building_ids.cat.categories.size

X_tab = pd.concat([df[num_features].astype(float), mgmt_dummies, cat_dummies], axis=1)
y_clf = df['is_delayed'].astype(int).values
y_reg = df['delay_hours'].astype(float).values

# Train/Val/Test split
idx = np.arange(len(df))
idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=42, stratify=y_clf)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y_clf[idx_temp])

scaler = StandardScaler()
X_tab_train = scaler.fit_transform(X_tab.values[idx_train])
X_tab_val   = scaler.transform(X_tab.values[idx_val])
X_tab_test  = scaler.transform(X_tab.values[idx_test])
bld_train, bld_val, bld_test = building_codes[idx_train], building_codes[idx_val], building_codes[idx_test]
y_clf_train, y_clf_val, y_clf_test = y_clf[idx_train], y_clf[idx_val], y_clf[idx_test]
y_reg_train, y_reg_val, y_reg_test = y_reg[idx_train], y_reg[idx_val], y_reg[idx_test]

# Regression target normalization
shift = abs(y_reg_train.min()) + 1
y_reg_train_t = np.log1p(y_reg_train + shift)
y_reg_val_t   = np.log1p(y_reg_val + shift)
y_reg_test_t  = np.log1p(y_reg_test + shift)
reg_mean, reg_std = y_reg_train_t.mean(), y_reg_train_t.std()
y_reg_train_n = (y_reg_train_t - reg_mean) / reg_std
y_reg_val_n   = (y_reg_val_t - reg_mean) / reg_std
y_reg_test_n  = (y_reg_test_t - reg_mean) / reg_std

print(f"Train/Val/Test: {len(idx_train):,} / {len(idx_val):,} / {len(idx_test):,}")

# ==========================================================================
# 3. NEW ARCHITECTURE — smaller, stronger regularization
# ==========================================================================
def build_ann_v2(task: str, n_tab: int, n_buildings: int, emb_dim: int = 4) -> Model:
    """Stronger regularization, smaller capacity, slower LR."""
    tab_in = layers.Input(shape=(n_tab,), name='tabular_input')
    bld_in = layers.Input(shape=(1,), name='building_input', dtype='int32')

    bld_emb = layers.Embedding(input_dim=n_buildings, output_dim=emb_dim,
                               embeddings_regularizer=tf.keras.regularizers.l2(1e-3),
                               name='building_embedding')(bld_in)
    bld_emb = layers.Flatten()(bld_emb)
    bld_emb = layers.Dropout(0.30)(bld_emb)        # NEW: dropout on embedding

    x = layers.Concatenate(name='concatenate')([tab_in, bld_emb])

    # Smaller architecture + stronger regularization
    for units, drop in [(256, 0.50), (128, 0.45), (64, 0.35),
                        (32, 0.25), (16, 0.20)]:
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)

    if task == 'classification':
        out = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss, metrics = 'binary_crossentropy', ['accuracy', tf.keras.metrics.AUC(name='auc')]
    else:
        out = layers.Dense(1, activation='linear', name='output')(x)
        loss, metrics = 'mse', ['mae']

    model = Model(inputs=[tab_in, bld_in], outputs=out, name=f'ANN_v2_{task}')
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),  # slower
                  loss=loss, metrics=metrics)
    return model

# ==========================================================================
# 4. TRAIN CLASSIFICATION
# ==========================================================================
print("\n" + "=" * 70)
print("TRAINING — CLASSIFICATION v2")
print("=" * 70)
model_clf = build_ann_v2('classification', X_tab_train.shape[1], n_buildings)
model_clf.summary(line_length=85)

es = callbacks.EarlyStopping(monitor='val_auc', mode='max',
                             patience=15, restore_best_weights=True, verbose=0)
rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                  min_lr=1e-5, verbose=0)

hist_clf = model_clf.fit(
    [X_tab_train, bld_train], y_clf_train,
    validation_data=([X_tab_val, bld_val], y_clf_val),
    epochs=100, batch_size=256, callbacks=[es, rlr], verbose=2
)

print("\n--- Classification Test Performance ---")
y_pred_proba = model_clf.predict([X_tab_test, bld_test], verbose=0).ravel()
y_pred_clf = (y_pred_proba >= 0.5).astype(int)
clf_metrics = {
    'Accuracy' : float(accuracy_score(y_clf_test, y_pred_clf)),
    'Precision': float(precision_score(y_clf_test, y_pred_clf)),
    'Recall'   : float(recall_score(y_clf_test, y_pred_clf)),
    'F1-score' : float(f1_score(y_clf_test, y_pred_clf)),
    'ROC-AUC'  : float(roc_auc_score(y_clf_test, y_pred_proba)),
}
for k, v in clf_metrics.items():
    print(f"  {k:10s}: {v:.4f}")

# Diagnose Train/Val gap
final_train_loss = hist_clf.history['loss'][-1]
final_val_loss = hist_clf.history['val_loss'][-1]
gap = final_val_loss - final_train_loss
print(f"\n  Train/Val Loss Gap (lower=better generalization): {gap:.4f}")

# ==========================================================================
# 5. TRAIN REGRESSION
# ==========================================================================
print("\n" + "=" * 70)
print("TRAINING — REGRESSION v2")
print("=" * 70)
model_reg = build_ann_v2('regression', X_tab_train.shape[1], n_buildings)
es_r = callbacks.EarlyStopping(monitor='val_loss', mode='min',
                               patience=15, restore_best_weights=True, verbose=0)
rlr_r = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                    min_lr=1e-5, verbose=0)

hist_reg = model_reg.fit(
    [X_tab_train, bld_train], y_reg_train_n,
    validation_data=([X_tab_val, bld_val], y_reg_val_n),
    epochs=100, batch_size=256, callbacks=[es_r, rlr_r], verbose=2
)

y_pred_reg_n = model_reg.predict([X_tab_test, bld_test], verbose=0).ravel()
y_pred_reg_t = y_pred_reg_n * reg_std + reg_mean
y_pred_reg = np.expm1(y_pred_reg_t) - shift

y_pred_val_n = model_reg.predict([X_tab_val, bld_val], verbose=0).ravel()
y_pred_val_t = y_pred_val_n * reg_std + reg_mean
y_pred_val = np.expm1(y_pred_val_t) - shift

reg_metrics = {
    'R²'    : float(r2_score(y_reg_test, y_pred_reg)),
    'MAE'   : float(mean_absolute_error(y_reg_test, y_pred_reg)),
    'RMSE'  : float(np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))),
    'Val R²': float(r2_score(y_reg_val, y_pred_val)),
}
print("\n--- Regression Test Performance ---")
for k, v in reg_metrics.items():
    print(f"  {k:6s}: {v:.4f}")

gap_r = hist_reg.history['val_loss'][-1] - hist_reg.history['loss'][-1]
print(f"\n  Train/Val Loss Gap: {gap_r:.4f}")

# ==========================================================================
# 6. SAVE
# ==========================================================================
model_clf.save('model_classification.keras')
model_reg.save('model_regression.keras')

import json
with open('model_metrics.json', 'w') as f:
    json.dump({'classification': clf_metrics,
               'regression': reg_metrics}, f, indent=2)

np.savez('predictions.npz',
         y_clf_test=y_clf_test, y_pred_proba=y_pred_proba, y_pred_clf=y_pred_clf,
         y_reg_test=y_reg_test, y_pred_reg=y_pred_reg,
         y_reg_val=y_reg_val, y_pred_val=y_pred_val,
         hist_clf_loss=hist_clf.history['loss'], hist_clf_val_loss=hist_clf.history['val_loss'],
         hist_clf_auc=hist_clf.history['auc'], hist_clf_val_auc=hist_clf.history['val_auc'],
         hist_reg_loss=hist_reg.history['loss'], hist_reg_val_loss=hist_reg.history['val_loss'])
print("\n✓ Saved models + metrics + predictions")
