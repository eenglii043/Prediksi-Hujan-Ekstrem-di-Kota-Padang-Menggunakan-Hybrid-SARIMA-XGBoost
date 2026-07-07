import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# ============================================
# STEP 19: SHAP ANALYSIS PADA MODEL FINAL (XGBoost V1)
# Menggunakan parameter terbaik yang sudah ditemukan di Step 15
# untuk memastikan model identik dengan yang dievaluasi sebelumnya.
# ============================================

train_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Train.csv"
test_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Test.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step19_shap_report.txt"
output_shap_values_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step19_shap_values.csv"
output_summary_plot = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step19_shap_summary_plot.png"
output_bar_plot = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step19_shap_bar_plot.png"
output_dependence_plot = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step19_shap_dependence_top_feature.png"

df_train = pd.read_csv(train_path, parse_dates=['TANGGAL']).sort_values('TANGGAL')
df_test = pd.read_csv(test_path, parse_dates=['TANGGAL']).sort_values('TANGGAL')

feature_cols = [c for c in df_train.columns if c not in ['TANGGAL', 'Extreme_Event']]

X_train = df_train[feature_cols]
y_train = df_train['Extreme_Event']
X_test = df_test[feature_cols]
y_test = df_test['Extreme_Event']

lines = []
lines.append("="*60)
lines.append("LAPORAN SHAP ANALYSIS - STEP 19")
lines.append("="*60)
lines.append(f"\nModel: XGBoost V1 (fitur asli, class_weight/scale_pos_weight)")
lines.append(f"Jumlah fitur: {len(feature_cols)}")

# ============================================
# RETRAIN MODEL DENGAN PARAMETER TERBAIK DARI STEP 15 (identik)
# ============================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

best_params = {
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 5,
    'n_estimators': 100,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'eval_metric': 'logloss'
}
lines.append(f"\nParameter model (identik dengan Step 15): {best_params}")

model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Validasi cepat bahwa model ini identik hasilnya dengan Step 15
from sklearn.metrics import recall_score, precision_score, f1_score
y_pred_check = model.predict(X_test)
lines.append(f"\n--- VALIDASI ULANG (harus sama dengan Step 15: Recall=0.5364, Precision=0.0846, F1=0.1462) ---")
lines.append(f"Recall    : {recall_score(y_test, y_pred_check):.4f}")
lines.append(f"Precision : {precision_score(y_test, y_pred_check):.4f}")
lines.append(f"F1-Score  : {f1_score(y_test, y_pred_check):.4f}")

# ============================================
# HITUNG SHAP VALUES (pada test set, agar mencerminkan performa generalisasi)
# ============================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

lines.append(f"\n--- SHAP VALUES DIHITUNG ---")
lines.append(f"Shape SHAP values: {shap_values.shape}")
lines.append(f"Base value (rata-rata prediksi log-odds sebelum fitur): {explainer.expected_value:.4f}")

# ============================================
# GLOBAL IMPORTANCE: rata-rata |SHAP value| per fitur
# ============================================
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'fitur': feature_cols,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

lines.append(f"\n--- GLOBAL FEATURE IMPORTANCE (rata-rata |SHAP value|) ---")
lines.append(shap_importance_df.to_string(index=False))

# ============================================
# ARAH PENGARUH: apakah fitur tinggi mendorong ke 'Ekstrem' atau 'Normal'
# ============================================
lines.append(f"\n--- ARAH PENGARUH FITUR TERATAS (korelasi nilai fitur vs SHAP value) ---")
top_features = shap_importance_df.head(8)['fitur'].tolist()
for feat in top_features:
    idx = feature_cols.index(feat)
    corr = np.corrcoef(X_test[feat].values, shap_values[:, idx])[0, 1]
    arah = "semakin TINGGI nilai fitur -> mendorong prediksi EKSTREM" if corr > 0 else "semakin TINGGI nilai fitur -> mendorong prediksi NORMAL"
    lines.append(f"{feat}: korelasi={corr:.3f} ({arah})")

# ============================================
# SIMPAN VISUALISASI SHAP
# ============================================
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig(output_summary_plot, dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(output_bar_plot, dpi=150, bbox_inches='tight')
plt.close()

# Dependence plot untuk fitur paling penting
top_feature = shap_importance_df.iloc[0]['fitur']
plt.figure()
shap.dependence_plot(top_feature, shap_values, X_test, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig(output_dependence_plot, dpi=150, bbox_inches='tight')
plt.close()

lines.append(f"\n--- VISUALISASI TERSIMPAN ---")
lines.append(f"Summary plot (beeswarm)  : {output_summary_plot}")
lines.append(f"Bar plot (global importance): {output_bar_plot}")
lines.append(f"Dependence plot ({top_feature}): {output_dependence_plot}")

# ============================================
# SIMPAN SHAP VALUES MENTAH KE CSV (untuk analisis lanjutan / lampiran skripsi)
# ============================================
shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{c}' for c in feature_cols])
shap_df.insert(0, 'TANGGAL', df_test['TANGGAL'].values)
shap_df.insert(1, 'y_true', y_test.values)
shap_df.insert(2, 'y_pred', y_pred_check)
shap_df.to_csv(output_shap_values_csv, index=False)

# Simpan laporan & ranking importance
with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
shap_importance_df.to_csv(r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step19_shap_importance_ranking.csv", index=False)

print(f"Selesai! Laporan: {output_report}")
print(f"\nTop 3 fitur paling berpengaruh:")
print(shap_importance_df.head(3).to_string(index=False))