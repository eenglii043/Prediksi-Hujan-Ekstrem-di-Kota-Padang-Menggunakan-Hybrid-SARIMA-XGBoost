import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

# ============================================
# STEP 18: THRESHOLD TUNING - MODEL 8 STASIUN BERSIH
# Cari titik threshold yang menyeimbangkan Recall tinggi dengan FP terkontrol
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_8Pos_Imputed_Clean.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step18_threshold_tuning_report.txt"

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

# Gunakan best params yang sama dari Step 17 (biar konsisten, tidak tuning ulang dari nol)
best_params = {'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1,
                'n_estimators': 100, 'scale_pos_weight': scale_pos_weight}

model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, **best_params)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]

lines = []
lines.append("="*60)
lines.append("LAPORAN THRESHOLD TUNING - STEP 18 (8 STASIUN BERSIH)")
lines.append("="*60)
lines.append(f"\nModel: XGBoost, params = {best_params}")
lines.append(f"Total kejadian ekstrem di test set: {int(y_test.sum())} dari {len(y_test)} baris")

lines.append(f"\n{'Threshold':<12}{'Recall':<10}{'Precision':<12}{'TP':<6}{'FP':<8}{'FN':<6}{'Total Alert':<12}{'F1':<8}")

results = []
for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    y_pred_t = (y_pred_proba >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    tn, fp, fn, tp = cm.ravel()
    rec = recall_score(y_test, y_pred_t)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    total_alert = tp + fp
    results.append({'threshold': t, 'recall': rec, 'precision': prec, 'tp': tp, 'fp': fp, 'fn': fn, 'total_alert': total_alert, 'f1': f1})
    lines.append(f"{t:<12}{rec:<10.4f}{prec:<12.4f}{tp:<6}{fp:<8}{fn:<6}{total_alert:<12}{f1:<8.4f}")

results_df = pd.DataFrame(results)

lines.append(f"\n--- ANALISIS PEMILIHAN THRESHOLD ---")
best_f1_row = results_df.loc[results_df['f1'].idxmax()]
lines.append(f"Threshold dengan F1 tertinggi: {best_f1_row['threshold']} (F1={best_f1_row['f1']:.4f}, Recall={best_f1_row['recall']:.4f})")

# Threshold dengan recall >= 0.70 tapi FP paling rendah (kandidat rekomendasi utk early warning)
candidates = results_df[results_df['recall'] >= 0.70]
if len(candidates) > 0:
    best_candidate = candidates.loc[candidates['fp'].idxmin()]
    lines.append(f"Threshold dengan Recall>=70% & FP minimal: {best_candidate['threshold']} (Recall={best_candidate['recall']:.4f}, FP={best_candidate['fp']:.0f})")
else:
    lines.append("Tidak ada threshold yang mencapai Recall >= 70% pada rentang yang diuji (0.30-0.65)")

lines.append(f"\nCatatan: pemilihan threshold final adalah keputusan kebijakan (trade-off keselamatan vs")
lines.append(f"alarm fatigue), bukan murni keputusan statistik. Rekomendasi: diskusikan dengan pihak")
lines.append(f"BMKG/BPBD mengenai toleransi mereka terhadap false alarm sebelum menetapkan threshold final.")

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Laporan: {output_report}")
print(results_df.to_string(index=False))