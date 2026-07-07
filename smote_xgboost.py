import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, classification_report,
                               recall_score, precision_score, f1_score, accuracy_score)

# ============================================
# STEP 18: SMOTE + XGBOOST
# SMOTE diterapkan HANYA pada data training (bukan test), untuk menghindari
# data leakage. Fitur yang dipakai: fitur V1 (asli), TANPA scale_pos_weight
# karena SMOTE sudah menyeimbangkan kelas secara langsung di training data.
# ============================================

train_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Train.csv"
test_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Test.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step18_smote_report.txt"
output_metrics_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step18_metrics_comparison.csv"
output_importance_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step18_feature_importance.csv"
output_predictions_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step18_predictions.csv"

df_train = pd.read_csv(train_path, parse_dates=['TANGGAL']).sort_values('TANGGAL')
df_test = pd.read_csv(test_path, parse_dates=['TANGGAL']).sort_values('TANGGAL')

feature_cols = [c for c in df_train.columns if c not in ['TANGGAL', 'Extreme_Event']]

X_train = df_train[feature_cols]
y_train = df_train['Extreme_Event']
X_test = df_test[feature_cols]
y_test = df_test['Extreme_Event']

lines = []
lines.append("="*60)
lines.append("LAPORAN SMOTE + XGBOOST - STEP 18")
lines.append("="*60)
lines.append(f"\nJumlah fitur: {len(feature_cols)}")
lines.append(f"Train SEBELUM SMOTE: {len(X_train)} baris")
lines.append(f"Distribusi kelas SEBELUM SMOTE: {y_train.value_counts().to_dict()}")

# ============================================
# TERAPKAN SMOTE (hanya di data training)
# ============================================
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

lines.append(f"\nTrain SETELAH SMOTE: {len(X_train_smote)} baris")
lines.append(f"Distribusi kelas SETELAH SMOTE: {y_train_smote.value_counts().to_dict()}")
lines.append(f"Jumlah data sintetis yang ditambahkan: {len(X_train_smote) - len(X_train)}")

# ============================================
# HYPERPARAMETER TUNING
# TANPA scale_pos_weight karena SMOTE sudah menyeimbangkan kelas
# NOTE: TimeSeriesSplit tidak ideal dipakai di atas data yang sudah di-SMOTE
# (karena urutan waktu jadi tidak relevan setelah oversampling), jadi
# untuk tuning kita pakai cross-validation biasa (StratifiedKFold via default)
# ============================================
from sklearn.model_selection import StratifiedKFold

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 5, 10]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lines.append(f"\n--- KONFIGURASI TUNING ---")
lines.append(f"Param grid: {param_grid}")
lines.append(f"Cross-validation: StratifiedKFold 5-fold (bukan TimeSeriesSplit, karena data sudah di-SMOTE)")
lines.append(f"Scoring: recall")

xgb_base = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
grid_search = GridSearchCV(xgb_base, param_grid, cv=skf, scoring='recall', n_jobs=-1, verbose=0)
grid_search.fit(X_train_smote, y_train_smote)

lines.append(f"\n--- HASIL TUNING ---")
lines.append(f"Best params: {grid_search.best_params_}")
lines.append(f"Best CV recall score: {grid_search.best_score_:.4f}")

best_xgb = grid_search.best_estimator_

# ============================================
# EVALUASI DI TEST SET ASLI (TIDAK di-SMOTE, harus data asli)
# ============================================
y_pred = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

lines.append(f"\n" + "="*60)
lines.append("HASIL EVALUASI XGBOOST + SMOTE DI TEST SET (data asli, tidak di-SMOTE)")
lines.append("="*60)
lines.append(f"\n--- CONFUSION MATRIX ---")
lines.append(f"                 Prediksi Normal   Prediksi Ekstrem")
lines.append(f"Aktual Normal    {cm[0][0]:<17} {cm[0][1]}")
lines.append(f"Aktual Ekstrem   {cm[1][0]:<17} {cm[1][1]}")
lines.append(f"\nAccuracy  : {acc:.4f}")
lines.append(f"Precision : {prec:.4f}")
lines.append(f"Recall    : {rec:.4f}")
lines.append(f"F1-Score  : {f1:.4f}")
lines.append(f"\n--- CLASSIFICATION REPORT ---")
lines.append(classification_report(y_test, y_pred, target_names=['Normal', 'Ekstrem']))

importance_df = pd.DataFrame({
    'fitur': feature_cols,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)
lines.append(f"\n--- FEATURE IMPORTANCE (XGBoost + SMOTE) ---")
lines.append(importance_df.to_string(index=False))

# ============================================
# PERBANDINGAN DENGAN XGBOOST V1 (tanpa SMOTE, Step 15)
# ============================================
comparison_df = pd.DataFrame([
    {'model': 'XGBoost V1 (class_weight, tanpa SMOTE)', 'accuracy': 0.6504, 'precision': 0.0846, 'recall': 0.5364, 'f1_score': 0.1462},
    {'model': 'XGBoost + SMOTE',                        'accuracy': acc,    'precision': prec,   'recall': rec,     'f1_score': f1},
])
lines.append(f"\n" + "="*60)
lines.append("PERBANDINGAN: XGBoost V1 (class_weight) vs XGBoost + SMOTE")
lines.append("="*60)
lines.append(comparison_df.to_string(index=False))

improvement = f1 - 0.1462
lines.append(f"\nPerubahan F1-Score vs V1: {improvement:+.4f} ({'membaik' if improvement > 0 else 'memburuk/stagnan'})")

# Simpan
with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
comparison_df.to_csv(output_metrics_csv, index=False)
importance_df.to_csv(output_importance_csv, index=False)

predictions_df = df_test[['TANGGAL']].copy()
predictions_df['y_true'] = y_test.values
predictions_df['y_pred'] = y_pred
predictions_df['y_pred_proba'] = y_pred_proba
predictions_df.to_csv(output_predictions_csv, index=False)

print(f"Selesai! Laporan: {output_report}")
print(f"Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")
print(f"Perubahan F1 vs V1: {improvement:+.4f}")