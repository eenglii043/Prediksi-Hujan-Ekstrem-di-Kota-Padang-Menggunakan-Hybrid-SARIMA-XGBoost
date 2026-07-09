import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle
import json

# ============================================
# STEP 20: LATIH & SIMPAN MODEL FINAL (8 STASIUN BERSIH)
# Output: model_xgboost_final_clean.pkl, feature_cols_final.json
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_8Pos_Imputed_Clean.csv"
output_model = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\model_xgboost_final_clean.pkl"
output_features = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\feature_cols_final.json"
output_neighbors = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\neighbor_map.json"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)
target_stations = ['Lubuk_Minturun', 'Nanggalo', 'Parak_Karakah']
all_stations = df.columns.tolist()  # 8 stasiun bersih

df['Bulan'] = df.index.month
df['Hari_Tahun'] = df.index.dayofyear
df['Hari_Minggu'] = df.index.dayofweek

for station in all_stations:
    df[f'{station}_lag1'] = df[station].shift(1)
    df[f'{station}_roll7'] = df[station].shift(1).rolling(7, min_periods=1).mean()

p95 = {s: df[s].quantile(0.95) for s in target_stations}

rows = []
neighbor_map = {}
for station in target_stations:
    neighbors = [s for s in all_stations if s != station]
    neighbor_map[station] = neighbors
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

feature_cols = ['Bulan', 'Hari_Tahun', 'Hari_Minggu', 'Lag1', 'Roll7', 'Tetangga_Lag1_avg', 'Tetangga_Roll7_avg']

# LATIH DI SELURUH DATA (bukan cuma train split) - karena ini untuk model produksi/dashboard,
# bukan untuk evaluasi. Evaluasi Recall/Precision yang kita laporkan tetap dari model
# yang dilatih di train split saja (Step 17-19) - JANGAN dicampur.
X_all = df_long[feature_cols]
y_all = df_long['Extreme_Event']
scale_pos_weight = (y_all == 0).sum() / (y_all == 1).sum()

best_params = {'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1,
                'n_estimators': 100, 'scale_pos_weight': scale_pos_weight}

model_production = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, **best_params)
model_production.fit(X_all, y_all)

with open(output_model, 'wb') as f:
    pickle.dump(model_production, f)

with open(output_features, 'w') as f:
    json.dump(feature_cols, f)

with open(output_neighbors, 'w') as f:
    json.dump(neighbor_map, f)

print(f"Model tersimpan: {output_model}")
print(f"Fitur tersimpan: {output_features}")
print(f"Neighbor map tersimpan: {output_neighbors}")
print(f"\nPENTING: model ini dilatih di SELURUH data (bukan cuma train split) untuk keperluan")
print(f"produksi/dashboard. Angka Recall/Precision yang dilaporkan di skripsi HARUS tetap")
print(f"dari model Step 17-19 yang dievaluasi di test set terpisah, BUKAN dari model ini.")