import pandas as pd
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                               confusion_matrix, precision_recall_curve)

# ============================================
# STEP 16: THRESHOLD TUNING UNTUK MODEL XGBOOST
# Default threshold 0.5 belum tentu optimal untuk kasus imbalance.
# Kita cari threshold yang memaksimalkan F1-Score (atau bisa juga
# diprioritaskan ke Recall, tergantung kebutuhan mitigasi bencana).
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step15_xgboost_predictions.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step16_threshold_tuning_report.txt"
output_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step16_threshold_comparison.csv"

df = pd.read_csv(input_path, parse_dates=['TANGGAL'])
y_true = df['y_true']
y_proba = df['y_pred_proba']

lines = []
lines.append("="*60)
lines.append("LAPORAN THRESHOLD TUNING - STEP 16")
lines.append("="*60)
lines.append(f"\nTotal data test: {len(df)}")
lines.append(f"Threshold default XGBoost (sebelumnya): 0.5")

# ============================================
# COBA BERBAGAI THRESHOLD (0.05 - 0.95)
# ============================================
thresholds = np.arange(0.05, 0.96, 0.05)
results = []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    
    # Handle kasus di mana tidak ada prediksi positif sama sekali
    if y_pred_t.sum() == 0:
        prec, rec, f1 = 0.0, 0.0, 0.0
    else:
        prec = precision_score(y_true, y_pred_t, zero_division=0)
        rec = recall_score(y_true, y_pred_t, zero_division=0)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    results.append({
        'threshold': round(t, 2),
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn,
        'true_negative': tn,
        'total_alert': tp + fp
    })

results_df = pd.DataFrame(results)

lines.append(f"\n--- TABEL PERBANDINGAN SEMUA THRESHOLD ---")
lines.append(results_df.to_string(index=False))

# ============================================
# CARI THRESHOLD OPTIMAL BERDASARKAN 2 KRITERIA
# ============================================
best_f1_row = results_df.loc[results_df['f1_score'].idxmax()]
# Threshold optimal untuk recall tinggi TAPI precision tidak terlalu jatuh
# (recall >= 0.5 dengan precision setinggi mungkin)
recall_priority = results_df[results_df['recall'] >= 0.5]
best_recall_priority_row = recall_priority.loc[recall_priority['precision'].idxmax()] if len(recall_priority) > 0 else None

lines.append(f"\n" + "="*60)
lines.append("REKOMENDASI THRESHOLD")
lines.append("="*60)

lines.append(f"\n--- Threshold terbaik berdasarkan F1-Score tertinggi ---")
lines.append(f"Threshold : {best_f1_row['threshold']}")
lines.append(f"Precision : {best_f1_row['precision']:.4f}")
lines.append(f"Recall    : {best_f1_row['recall']:.4f}")
lines.append(f"F1-Score  : {best_f1_row['f1_score']:.4f}")
lines.append(f"Total Alert (TP+FP) : {int(best_f1_row['total_alert'])}")

if best_recall_priority_row is not None:
    lines.append(f"\n--- Threshold terbaik untuk mitigasi bencana (Recall >= 0.5, Precision maksimal) ---")
    lines.append(f"Threshold : {best_recall_priority_row['threshold']}")
    lines.append(f"Precision : {best_recall_priority_row['precision']:.4f}")
    lines.append(f"Recall    : {best_recall_priority_row['recall']:.4f}")
    lines.append(f"F1-Score  : {best_recall_priority_row['f1_score']:.4f}")
    lines.append(f"Total Alert (TP+FP) : {int(best_recall_priority_row['total_alert'])}")
else:
    lines.append(f"\nTidak ada threshold yang mencapai Recall >= 0.5 selain default")

# Bandingkan dengan threshold default 0.5
default_row = results_df[results_df['threshold'] == 0.5]
lines.append(f"\n--- PEMBANDING: Threshold default (0.5) ---")
if len(default_row) > 0:
    lines.append(default_row.to_string(index=False))

# ============================================
# CATATAN INTERPRETASI
# ============================================
lines.append(f"\n" + "="*60)
lines.append("CATATAN INTERPRETASI")
lines.append("="*60)
lines.append("""
Threshold lebih RENDAH dari 0.5 -> model lebih 'gampang curiga' (semua nilai
probabilitas rendah pun dianggap ekstrem) -> Recall naik, tapi Precision turun
(banyak alarm palsu).

Threshold lebih TINGGI dari 0.5 -> model lebih 'pemilih' (hanya nilai
probabilitas tinggi yang dianggap ekstrem) -> Precision naik, tapi Recall
turun (banyak kejadian ekstrem asli malah terlewat/miss).

Untuk konteks mitigasi bencana, biasanya threshold yang SEDIKIT lebih rendah
dari 0.5 lebih disukai (mengutamakan Recall), karena False Negative (model
bilang 'aman' padahal ekstrem) jauh lebih berbahaya dari False Positive
(model bilang 'waspada' padahal ternyata aman).
""")

# Simpan
with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

results_df.to_csv(output_csv, index=False)

print(f"Selesai! Laporan tersimpan di: {output_report}")
print(f"Tabel perbandingan tersimpan di: {output_csv}")
print(f"\nThreshold terbaik (F1 max): {best_f1_row['threshold']} -> F1={best_f1_row['f1_score']:.4f}")
if best_recall_priority_row is not None:
    print(f"Threshold terbaik (Recall>=0.5): {best_recall_priority_row['threshold']} -> Precision={best_recall_priority_row['precision']:.4f}, Recall={best_recall_priority_row['recall']:.4f}")