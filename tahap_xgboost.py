import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, classification_report,
                               recall_score, precision_score, f1_score, accuracy_score)

# ============================================
# STEP 15: XGBOOST + HYPERPARAMETER TUNING
# Model gradient boosting - biasanya lebih kuat menangkap pola non-linear
# dibanding Random Forest untuk data tabular dengan sinyal lemah
# ============================================

train_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Train.csv"
test_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Test.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step15_xgboost_report.txt"
output_metrics_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step15_xgboost_metrics.csv"
output_importance_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step15_xgboost_feature_importance.csv"
output_predictions_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step15_xgboost_predictions.csv"

df_train = pd.read_csv(train_path, parse_dates=['TANGGAL']).sort_values('TANGGAL')
df_test = pd.read_csv(test_path, parse_dates=['TANGGAL']).sort_values('TANGGAL')

feature_cols = [c for c in df_train.columns if c not in ['TANGGAL', 'Extreme_Event']]

X_train = df_train[feature_cols]
y_train = df_train['Extreme_Event']
X_test = df_test[feature_cols]
y_test = df_test['Extreme_Event']

lines = []
lines.append("="*60)
lines.append("LAPORAN XGBOOST - STEP 15")
lines.append("="*60)
lines.append(f"\nJumlah fitur: {len(feature_cols)}")
lines.append(f"Train: {len(X_train)} baris | Test: {len(X_test)} baris")

# ============================================
# HYPERPARAMETER TUNING
# scale_pos_weight menangani class imbalance (rasio negatif:positif)
# ============================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
lines.append(f"\nscale_pos_weight dihitung dari train: {scale_pos_weight:.2f}")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 5, 10],
    'scale_pos_weight': [1, scale_pos_weight]  # bandingkan tanpa & dengan penanganan imbalance
}

tscv = TimeSeriesSplit(n_splits=5)

lines.append(f"\n--- KONFIGURASI TUNING ---")
lines.append(f"Param grid: {param_grid}")
lines.append(f"Cross-validation: TimeSeriesSplit (5 fold, menjaga urutan waktu)")
lines.append(f"Scoring: recall (prioritas utama untuk mitigasi bencana)")

xgb_base = XGBClassifier(
    random_state=42, eval_metric='logloss', n_jobs=-1
)
grid_search = GridSearchCV(xgb_base, param_grid, cv=tscv, scoring='recall', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

lines.append(f"\n--- HASIL TUNING ---")
lines.append(f"Best params: {grid_search.best_params_}")
lines.append(f"Best CV recall score: {grid_search.best_score_:.4f}")

best_xgb = grid_search.best_estimator_

# ============================================
# EVALUASI DI TEST SET
# ============================================
y_pred = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
lines.append(f"\n" + "="*60)
lines.append("HASIL EVALUASI XGBOOST (TUNED) DI TEST SET")
lines.append("="*60)
lines.append(f"\n--- CONFUSION MATRIX ---")
lines.append(f"                 Prediksi Normal   Prediksi Ekstrem")
lines.append(f"Aktual Normal    {cm[0][0]:<17} {cm[0][1]}")
lines.append(f"Aktual Ekstrem   {cm[1][0]:<17} {cm[1][1]}")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

lines.append(f"\n--- METRIK UTAMA ---")
lines.append(f"Accuracy  : {acc:.4f}")
lines.append(f"Precision : {prec:.4f}")
lines.append(f"Recall    : {rec:.4f}")
lines.append(f"F1-Score  : {f1:.4f}")

lines.append(f"\n--- CLASSIFICATION REPORT LENGKAP ---")
lines.append(classification_report(y_test, y_pred, target_names=['Normal', 'Ekstrem']))

# Feature importance
importance_df = pd.DataFrame({
    'fitur': feature_cols,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)
lines.append(f"\n--- FEATURE IMPORTANCE (XGBoost) ---")
lines.append(importance_df.to_string(index=False))

# ============================================
# PERBANDINGAN 3 MODEL SEKALIGUS
# ============================================
comparison_df = pd.DataFrame([
    {'model': 'Logistic Regression', 'accuracy': 0.7230, 'precision': 0.0918, 'recall': 0.4455, 'f1_score': 0.1522},
    {'model': 'Random Forest',       'accuracy': 0.6687, 'precision': 0.0804, 'recall': 0.4727, 'f1_score': 0.1374},
    {'model': 'XGBoost',             'accuracy': acc,    'precision': prec,   'recall': rec,     'f1_score': f1},
])

lines.append(f"\n" + "="*60)
lines.append("PERBANDINGAN 3 MODEL")
lines.append("="*60)
lines.append(comparison_df.to_string(index=False))

best_model_name = comparison_df.loc[comparison_df['f1_score'].idxmax(), 'model']
lines.append(f"\n>>> MODEL TERBAIK BERDASARKAN F1-SCORE: {best_model_name} <<<")

# ============================================
# SIMPAN SEMUA OUTPUT
# ============================================
# 1. Laporan teks lengkap
with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# 2. Metrik perbandingan 3 model (CSV)
comparison_df.to_csv(output_metrics_csv, index=False)

# 3. Feature importance XGBoost (CSV)
importance_df.to_csv(output_importance_csv, index=False)

# 4. Prediksi detail per baris test set (CSV) - berguna untuk analisis kesalahan nanti
predictions_df = df_test[['TANGGAL']].copy()
predictions_df['y_true'] = y_test.values
predictions_df['y_pred'] = y_pred
predictions_df['y_pred_proba'] = y_pred_proba
predictions_df.to_csv(output_predictions_csv, index=False)

print(f"Selesai! 4 file tersimpan:")
print(f"  - Laporan lengkap : {output_report}")
print(f"  - Metrik 3 model  : {output_metrics_csv}")
print(f"  - Feature importance : {output_importance_csv}")
print(f"  - Prediksi detail : {output_predictions_csv}")
print(f"\nBest params: {grid_search.best_params_}")
print(f"Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")
print(f"Model terbaik (F1): {best_model_name}")