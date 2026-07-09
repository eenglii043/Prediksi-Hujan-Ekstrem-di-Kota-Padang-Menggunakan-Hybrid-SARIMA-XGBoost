import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# ============================================
# STEP 19: SHAP INTERPRETATION - MODEL FINAL (8 STASIUN BERSIH)
# Model: XGBoost, threshold 0.40, Recall=0.8545, Precision=0.0706
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_8Pos_Imputed_Clean.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step19_shap_report.txt"
output_plot_summary = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\shap_summary_plot.png"
output_plot_bar = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\shap_bar_plot.png"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)
target_stations = ['Lubuk_Minturun', 'Nanggalo', 'Parak_Karakah']
all_stations = df.columns.tolist()

df['Bulan'] = df.index.month
df['Hari_Tahun'] = df.index.dayofyear
df['Hari_Minggu'] = df.index.dayofweek

for station in all_stations:
    df[f'{station}_lag1'] = df[station].shift(1)
    df[f'{station}_roll7'] = df[station].shift(1).rolling(7, min_periods=1).mean()

p95 = {s: df[s].quantile(0.95) for s in target_stations}

rows = []
for station in target_stations:
    neighbors = [s for s in all_stations if s != station]
    temp = pd.DataFrame(index=df.index)
    temp.index.name = None
    temp['TANGGAL'] = df.index
    temp['STASIUN'] = station
    temp['Bulan'] = df['Bulan']
    temp['Hari_Tahun'] = df['Hari_Tahun']
    temp['Hari_Minggu'] = df['Hari_Minggu']
    temp['Lag1'] = df[f'{station}_lag1']
    temp['Roll7'] = df[f'{station}_roll7']
    temp['Tetangga_Lag1_avg'] = df[[f'{n}_lag1' for n in neighbors]].mean(axis=1)
    temp['Tetangga_Roll7_avg'] = df[[f'{n}_roll7' for n in neighbors]].mean(axis=1)
    temp['Extreme_Event'] = (df[station] > p95[station]).astype(int)
    rows.append(temp)

df_long = pd.concat(rows, axis=0).sort_values(['TANGGAL', 'STASIUN']).reset_index(drop=True)
df_long = df_long.dropna()

unique_dates = sorted(df_long['TANGGAL'].unique())
split_date = unique_dates[int(len(unique_dates) * 0.8)]
df_train = df_long[df_long['TANGGAL'] < split_date].copy()
df_test = df_long[df_long['TANGGAL'] >= split_date].copy()

feature_cols = ['Bulan', 'Hari_Tahun', 'Hari_Minggu', 'Lag1', 'Roll7', 'Tetangga_Lag1_avg', 'Tetangga_Roll7_avg']
X_train, y_train = df_train[feature_cols], df_train['Extreme_Event']
X_test, y_test = df_test[feature_cols], df_test['Extreme_Event']
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

best_params = {'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1,
                'n_estimators': 100, 'scale_pos_weight': scale_pos_weight}

model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, **best_params)
model.fit(X_train, y_train)

lines = []
lines.append("="*60)
lines.append("LAPORAN SHAP INTERPRETATION - STEP 19 (MODEL FINAL)")
lines.append("="*60)
lines.append(f"\nModel: XGBoost, 8 stasiun bersih, {len(feature_cols)} fitur")
lines.append(f"Fitur: {feature_cols}")

# ============================================
# SHAP TreeExplainer (native untuk model tree-based, cepat & eksak)
# ============================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance (mean absolute SHAP value)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'fitur': feature_cols,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)
importance_df['persentase'] = importance_df['mean_abs_shap'] / importance_df['mean_abs_shap'].sum() * 100

lines.append(f"\n--- RANKING FITUR BERDASARKAN SHAP (mean |SHAP value|) ---")
lines.append(importance_df.to_string(index=False))

# Arah pengaruh: korelasi antara nilai fitur asli dengan SHAP value-nya
lines.append(f"\n--- ARAH PENGARUH TIAP FITUR (korelasi nilai fitur vs SHAP value) ---")
for i, col in enumerate(feature_cols):
    corr = np.corrcoef(X_test[col].values, shap_values[:, i])[0, 1]
    arah = "-> Naik = lebih EKSTREM" if corr > 0 else "-> Naik = lebih NORMAL"
    lines.append(f"{col}: korelasi={corr:.4f} {arah}")

# Simpan plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(output_plot_summary, dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(output_plot_bar, dpi=150, bbox_inches='tight')
plt.close()

lines.append(f"\n--- PLOT DISIMPAN ---")
lines.append(f"Summary plot (beeswarm): {output_plot_summary}")
lines.append(f"Bar plot (ranking): {output_plot_bar}")

# Perbandingan dengan klaim SHAP lama (dari dokumen DeepSeek)
lines.append(f"\n--- PERBANDINGAN DENGAN KLAIM SEBELUMNYA (data 9/10 stasiun bermasalah) ---")
lines.append(f"{'Fitur':<25}{'Klaim lama (%)':<18}{'Hasil bersih (%)':<18}")
old_claims = {'Tetangga_Lag1_avg': 28.27, 'Lag2': 9.2, 'Lag1': 8.2, 'Hari_Minggu': 8.2, 'Hari_Tahun': 8.1}
for fitur, old_pct in old_claims.items():
    new_row = importance_df[importance_df['fitur'] == fitur]
    new_pct = new_row['persentase'].values[0] if len(new_row) > 0 else None
    new_str = f"{new_pct:.2f}" if new_pct is not None else "N/A (fitur beda)"
    lines.append(f"{fitur:<25}{old_pct:<18}{new_str:<18}")

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Laporan: {output_report}")
print(importance_df.to_string(index=False))