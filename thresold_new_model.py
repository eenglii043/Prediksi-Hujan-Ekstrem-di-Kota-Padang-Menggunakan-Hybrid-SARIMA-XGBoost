import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step21_threshold_tuning_new_model.txt"
output_lines = []

def log(text):
    print(text)
    output_lines.append(str(text))

try:
    # =========================================================
    # 1. LOAD & PREPARE SAMA SEPERTI SEBELUMNYA
    # =========================================================
    input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Padang_Filtered.csv"
    df_raw = pd.read_csv(input_path, parse_dates=['DATA TIMESTAMP'])
    df_raw['TANGGAL'] = pd.to_datetime(df_raw['DATA TIMESTAMP'].dt.date)

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

    df_selected['RAINFALL DAY MM'] = df_selected['RAINFALL DAY MM'].replace([9999, 9999.0], np.nan)
    df_selected['RAINFALL DAY MM'] = df_selected['RAINFALL DAY MM'].replace([8888, 8888.0], 0)

    df_selected['STASIUN'] = df_selected['POS HUJAN ID'].map(lambda x: target_names.get(x, x))
    df_wide = df_selected.pivot_table(index='TANGGAL', columns='STASIUN', values='RAINFALL DAY MM', aggfunc='mean')
    full_range = pd.date_range('2017-01-01', '2025-12-31', freq='D')
    df_wide = df_wide.reindex(full_range)
    df_wide.index.name = 'TANGGAL'

    knn = KNNImputer(n_neighbors=5, weights='distance')
    df_imp = pd.DataFrame(knn.fit_transform(df_wide), index=df_wide.index, columns=df_wide.columns)
    df_imp = df_imp.clip(lower=0)

    # =========================================================
    # 2. FEATURE ENGINEERING
    # =========================================================
    target_cols = ['Parak_Karakah', 'Nanggalo', 'Lubuk_Minturun']
    all_cols = df_imp.columns.tolist()
    other_cols = [c for c in all_cols if c not in target_cols]

    df_feat = df_imp.copy()
    df_feat['Bulan'] = df_feat.index.month
    df_feat['Hari_Tahun'] = df_feat.index.dayofyear
    df_feat['Hari_Minggu'] = df_feat.index.dayofweek

    for col in all_cols:
        for lag in [1,2,3]:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)
        df_feat[f'{col}_roll7'] = df_feat[col].shift(1).rolling(7, min_periods=1).mean()
        df_feat[f'{col}_roll30'] = df_feat[col].shift(1).rolling(30, min_periods=1).mean()

    for st in target_cols:
        df_feat[f'{st}_Tetangga_Lag1_avg'] = df_feat[[f'{o}_lag1' for o in other_cols]].mean(axis=1)
        df_feat[f'{st}_Tetangga_Roll7_avg'] = df_feat[[f'{o}_roll7' for o in other_cols]].mean(axis=1)

    # =========================================================
    # 3. RESHAPE & SPLIT
    # =========================================================
    rows = []
    for st in target_cols:
        temp = pd.DataFrame(index=df_feat.index)
        temp['TANGGAL'] = df_feat.index
        temp['STASIUN'] = st
        temp['Bulan'] = df_feat['Bulan']
        temp['Hari_Tahun'] = df_feat['Hari_Tahun']
        temp['Hari_Minggu'] = df_feat['Hari_Minggu']
        for lag in [1,2,3]:
            temp[f'Lag{lag}'] = df_feat[f'{st}_lag{lag}']
        temp['Roll7'] = df_feat[f'{st}_roll7']
        temp['Roll30'] = df_feat[f'{st}_roll30']
        temp['Tetangga_Lag1_avg'] = df_feat[f'{st}_Tetangga_Lag1_avg']
        temp['Tetangga_Roll7_avg'] = df_feat[f'{st}_Tetangga_Roll7_avg']
        temp['_Curah_Hujan_Hari_Ini_REF'] = df_feat[st]
        p95 = df_feat[st].quantile(0.95)
        temp['Extreme_Event'] = (df_feat[st] > p95).astype(int)
        rows.append(temp)

    df_long = pd.concat(rows, axis=0).reset_index(drop=True)
    df_long = df_long.dropna()

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

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================
    # 4. TRAIN XGBOOST
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
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    log("="*60)
    log("THRESHOLD TUNING - MODEL SPASIAL 9 STASIUN")
    log("="*60)
    log(f"Total data test: {len(y_test)}")
    log(f"Jumlah kejadian ekstrem aktual: {y_test.sum()}")

    # =========================================================
    # 5. COBA THRESHOLD 0.05 - 0.95
    # =========================================================
    thresholds = np.arange(0.05, 0.96, 0.05)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if y_pred.sum() == 0:
            rec = 0.0
            prec = 0.0
            f1 = 0.0
            tp = 0
            fp = 0
            fn = 0
        else:
            rec = recall_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            tp = cm[1,1]
            fp = cm[0,1]
            fn = cm[1,0]
        results.append({'threshold': t, 'recall': rec, 'precision': prec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'total_alert': tp+fp})

    df_results = pd.DataFrame(results)

    # Cari threshold terbaik berdasarkan F1
    best_f1 = df_results.loc[df_results['f1'].idxmax()]
    # Cari threshold dengan recall tertinggi di atas 0.5
    best_recall = df_results[df_results['recall'] >= 0.5].iloc[-1] if len(df_results[df_results['recall'] >= 0.5]) > 0 else None

    log("\n--- TABEL PERBANDINGAN THRESHOLD ---")
    log(df_results.to_string(index=False))
    log("\n" + "="*60)
    log("REKOMENDASI")
    log("="*60)
    log(f"Threshold terbaik (F1 max): {best_f1['threshold']} -> Recall={best_f1['recall']:.4f}, Precision={best_f1['precision']:.4f}, F1={best_f1['f1']:.4f} (TP={int(best_f1['tp'])}, FP={int(best_f1['fp'])})")
    
    if best_recall is not None:
        log(f"Threshold terbaik (Recall >= 0.5): {best_recall['threshold']} -> Recall={best_recall['recall']:.4f}, Precision={best_recall['precision']:.4f}, F1={best_recall['f1']:.4f} (TP={int(best_recall['tp'])}, FP={int(best_recall['fp'])})")
    
    # Default 0.5
    default = df_results[df_results['threshold'] == 0.5]
    if not default.empty:
        log(f"\nThreshold default 0.5: Recall={default['recall'].values[0]:.4f}, Precision={default['precision'].values[0]:.4f}, F1={default['f1'].values[0]:.4f}")

    log("\nCatatan: Threshold lebih rendah = lebih banyak alarm palsu tapi recall lebih tinggi.")
    log("Untuk mitigasi bencana, prioritas Recall lebih penting (lebih baik banyak alarm daripada kelewat).")

except Exception as e:
    import traceback
    log(traceback.format_exc())

finally:
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\nLaporan tersimpan di: {output_report}")