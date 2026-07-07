import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import traceback
import sys

# =========================================================
# PATH OUTPUT REPORT
# =========================================================
output_report_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step20_spatial_expansion_report.txt"

# =========================================================
# FUNGSI UNTUK MENAMPUNG OUTPUT
# =========================================================
output_lines = []

def log(text):
    print(text)
    output_lines.append(str(text))

try:
    # =========================================================
    # 1. LOAD DATA 9 STASIUN KOTA PADANG
    # =========================================================
    input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Padang_Filtered.csv"
    df_raw = pd.read_csv(input_path, parse_dates=['DATA TIMESTAMP'])
    df_raw['TANGGAL'] = pd.to_datetime(df_raw['DATA TIMESTAMP'].dt.date)

    # =========================================================
    # 2. TARGET & OTHER STASIUN
    # =========================================================
    target_ids = ['13710202a', '13711001a', '13711102a']
    target_names = {
        '13710202a': 'Parak_Karakah',
        '13711001a': 'Nanggalo',
        '13711102a': 'Lubuk_Minturun'
    }

    all_stations = df_raw['POS HUJAN ID'].unique()
    other_ids = [s for s in all_stations if s.startswith('1371') and s not in target_ids]
    selected_ids = target_ids + other_ids
    df_selected = df_raw[df_raw['POS HUJAN ID'].isin(selected_ids)].copy()

    log("="*60)
    log("LAPORAN EKSPANSI SPASIAL - 9 STASIUN KOTA PADANG")
    log("="*60)
    log(f"Total stasiun Kota Padang: {len(all_stations)}")
    log(f"Target (3): {target_ids}")
    log(f"Tambahan ({len(other_ids)}): {other_ids}")

    # Cleaning sentinel
    df_selected['RAINFALL DAY MM'] = df_selected['RAINFALL DAY MM'].replace([9999, 9999.0], np.nan)
    df_selected['RAINFALL DAY MM'] = df_selected['RAINFALL DAY MM'].replace([8888, 8888.0], 0)

    # Pivot ke wide
    df_selected['STASIUN'] = df_selected['POS HUJAN ID'].map(lambda x: target_names.get(x, x))
    df_wide = df_selected.pivot_table(index='TANGGAL', columns='STASIUN', values='RAINFALL DAY MM', aggfunc='mean')
    full_range = pd.date_range('2017-01-01', '2025-12-31', freq='D')
    df_wide = df_wide.reindex(full_range)
    df_wide.index.name = 'TANGGAL'

    log(f"Shape wide (9 stasiun): {df_wide.shape}")

    # Imputasi KNN
    knn = KNNImputer(n_neighbors=5, weights='distance')
    df_imp = pd.DataFrame(knn.fit_transform(df_wide), index=df_wide.index, columns=df_wide.columns)
    df_imp = df_imp.clip(lower=0)
    log("Imputasi KNN selesai. NaN tersisa: 0")

    # =========================================================
    # 3. FEATURE ENGINEERING UNTUK SEMUA STASIUN
    # =========================================================
    target_cols = ['Parak_Karakah', 'Nanggalo', 'Lubuk_Minturun']
    all_cols = df_imp.columns.tolist()
    other_cols = [c for c in all_cols if c not in target_cols]

    df_feat = df_imp.copy()

    # Fitur waktu
    df_feat['Bulan'] = df_feat.index.month
    df_feat['Hari_Tahun'] = df_feat.index.dayofyear
    df_feat['Hari_Minggu'] = df_feat.index.dayofweek

    # Buat lag & rolling untuk SEMUA stasiun
    for col in all_cols:
        for lag in [1,2,3]:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)
        df_feat[f'{col}_roll7'] = df_feat[col].shift(1).rolling(7, min_periods=1).mean()
        df_feat[f'{col}_roll30'] = df_feat[col].shift(1).rolling(30, min_periods=1).mean()

    # =========================================================
    # 4. BUAT FITUR SPASIAL UNTUK MASING-MASING TARGET
    # =========================================================
    for st in target_cols:
        df_feat[f'{st}_Tetangga_Lag1_avg'] = df_feat[[f'{o}_lag1' for o in other_cols]].mean(axis=1)
        df_feat[f'{st}_Tetangga_Roll7_avg'] = df_feat[[f'{o}_roll7' for o in other_cols]].mean(axis=1)

    # =========================================================
    # 5. RESHAPE KE LONG FORMAT
    # =========================================================
    rows = []
    for st in target_cols:
        temp = pd.DataFrame(index=df_feat.index)
        temp['TANGGAL'] = df_feat.index
        temp['STASIUN'] = st
        # Waktu
        temp['Bulan'] = df_feat['Bulan']
        temp['Hari_Tahun'] = df_feat['Hari_Tahun']
        temp['Hari_Minggu'] = df_feat['Hari_Minggu']
        # Fitur historis sendiri
        for lag in [1,2,3]:
            temp[f'Lag{lag}'] = df_feat[f'{st}_lag{lag}']
        temp['Roll7'] = df_feat[f'{st}_roll7']
        temp['Roll30'] = df_feat[f'{st}_roll30']
        # Fitur spasial
        temp['Tetangga_Lag1_avg'] = df_feat[f'{st}_Tetangga_Lag1_avg']
        temp['Tetangga_Roll7_avg'] = df_feat[f'{st}_Tetangga_Roll7_avg']
        # Referensi (tidak jadi fitur)
        temp['_Curah_Hujan_Hari_Ini_REF'] = df_feat[st]
        # Target (persentil-95)
        p95 = df_feat[st].quantile(0.95)
        temp['Extreme_Event'] = (df_feat[st] > p95).astype(int)
        rows.append(temp)

    df_long = pd.concat(rows, axis=0).reset_index(drop=True)
    df_long = df_long.sort_values(['TANGGAL', 'STASIUN']).reset_index(drop=True)

    # Drop baris dengan NaN (akibat lag/rolling di awal)
    before_drop = len(df_long)
    df_long = df_long.dropna()
    after_drop = len(df_long)
    log(f"Baris sebelum drop NaN: {before_drop}")
    log(f"Baris setelah drop NaN: {after_drop}")
    log(f"Shape long format (3 stasiun x hari): {df_long.shape}")

    if df_long.empty:
        raise ValueError("Data kosong setelah dropna, periksa feature engineering.")

    # =========================================================
    # 6. ENCODING & SPLIT TIME-BASED
    # =========================================================
    df_encoded = pd.get_dummies(df_long, columns=['STASIUN'], drop_first=False)
    unique_dates = sorted(df_encoded['TANGGAL'].unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    df_train = df_encoded[df_encoded['TANGGAL'] < split_date].copy()
    df_test = df_encoded[df_encoded['TANGGAL'] >= split_date].copy()

    exclude_cols = ['TANGGAL', '_Curah_Hujan_Hari_Ini_REF', 'Extreme_Event']
    feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]

    X_train = df_train[feature_cols]
    y_train = df_train['Extreme_Event']
    X_test = df_test[feature_cols]
    y_test = df_test['Extreme_Event']

    log(f"Split date: {split_date.date()}")
    log(f"Train size: {len(X_train)} baris")
    log(f"Test size: {len(X_test)} baris")
    log(f"Distribusi target train: {y_train.value_counts().to_dict()}")
    log(f"Distribusi target test: {y_test.value_counts().to_dict()}")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================
    # 7. TRAIN XGBOOST
    # =========================================================
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.01,
        'min_child_weight': 5,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': -1
    }
    model = XGBClassifier(**params)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    log("\n" + "="*60)
    log("HASIL XGBOOST DENGAN SPASIAL 9 STASIUN")
    log("="*60)
    log(f"Confusion Matrix:")
    log(f"  Prediksi Normal  Prediksi Ekstrem")
    log(f"Aktual Normal   {cm[0][0]:<10}     {cm[0][1]}")
    log(f"Aktual Ekstrem  {cm[1][0]:<10}     {cm[1][1]}")
    log(f"\nAccuracy : {acc:.4f}")
    log(f"Precision: {prec:.4f}")
    log(f"Recall   : {rec:.4f}")
    log(f"F1-Score : {f1:.4f}")

    # Bandingkan dengan model lama
    old_recall = 0.5364
    old_prec = 0.0846
    old_f1 = 0.1462
    log("\n" + "="*60)
    log("PERBANDINGAN DENGAN MODEL LAMA (3 stasiun, 2 tetangga)")
    log("="*60)
    log(f"{'Metrik':<15}{'Model Lama':<18}{'Model Baru (9 stasiun)':<18}")
    log(f"{'Recall':<15}{old_recall:<18.4f}{rec:<18.4f}")
    log(f"{'Precision':<15}{old_prec:<18.4f}{prec:<18.4f}")
    log(f"{'F1-Score':<15}{old_f1:<18.4f}{f1:<18.4f}")
    log(f"\nPerubahan Recall: {rec - old_recall:+.4f}")

    if rec > old_recall:
        log(">>> RECALL MENINGKAT! <<<")
    else:
        log(">>> RECALL TIDAK MENINGKAT (atau tetap) <<<")

except Exception as e:
    log("\n" + "="*60)
    log("ERROR TERJADI:")
    log("="*60)
    log(traceback.format_exc())

finally:
    # =========================================================
    # 8. SIMPAN SEMUA OUTPUT KE FILE TXT
    # =========================================================
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    log(f"\nLaporan lengkap tersimpan di: {output_report_path}")
    print("Proses selesai.")