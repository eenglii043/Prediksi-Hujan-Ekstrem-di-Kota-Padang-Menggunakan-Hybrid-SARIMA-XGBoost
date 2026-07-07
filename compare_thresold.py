import pandas as pd
import numpy as np

# ============================================
# STEP 10: BANDINGKAN 3 DEFINISI THRESHOLD EKSTREM
# Opsi A: Persentil ke-95 (per stasiun, adaptif terhadap karakteristik lokal)
# Opsi B: 150mm (kategori "Hujan Sangat Lebat" BMKG)
# Opsi C: 50mm (ambang pemicu banjir, ref. Dhuha dkk. 2024 / skripsi Atira)
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Features.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step10_threshold_comparison_report.txt"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)
stations = ['Lubuk_Minturun', 'Nanggalo', 'Parak_Karakah']

lines = []
lines.append("="*60)
lines.append("LAPORAN PERBANDINGAN THRESHOLD EKSTREM - STEP 10")
lines.append("="*60)

lines.append(f"\nTotal hari yang dianalisis: {len(df)}")

# ============================================
# HITUNG PERSENTIL KE-95 PER STASIUN (dari hari dengan hujan > 0 saja, lebih representatif)
# ============================================
lines.append("\n--- NILAI PERSENTIL KE-95 PER STASIUN ---")
p95_values = {}
for station in stations:
    p95_all = df[station].quantile(0.95)
    p95_values[station] = p95_all
    lines.append(f"{station}: persentil-95 dari seluruh data = {p95_all:.2f}mm")

lines.append("\n(Untuk perbandingan, persentil-95 hanya dari hari hujan (>0mm), lebih representatif kejadian ekstrem):")
p95_rainonly = {}
for station in stations:
    rain_days = df[df[station] > 0][station]
    p95_ro = rain_days.quantile(0.95)
    p95_rainonly[station] = p95_ro
    lines.append(f"{station}: persentil-95 (hari hujan saja) = {p95_ro:.2f}mm")

# ============================================
# BUAT 3 VERSI LABEL & HITUNG DISTRIBUSINYA
# ============================================
lines.append("\n" + "="*60)
lines.append("PERBANDINGAN DISTRIBUSI LABEL PER OPSI THRESHOLD")
lines.append("="*60)

results_summary = []

for station in stations:
    lines.append(f"\n{'='*40}")
    lines.append(f"STASIUN: {station}")
    lines.append(f"{'='*40}")

    # Opsi A: Persentil-95 (dari seluruh data termasuk 0)
    label_p95 = (df[station] > p95_values[station]).astype(int)
    n_extreme_p95 = label_p95.sum()
    pct_p95 = n_extreme_p95 / len(df) * 100

    # Opsi B: 150mm
    label_150 = (df[station] > 150).astype(int)
    n_extreme_150 = label_150.sum()
    pct_150 = n_extreme_150 / len(df) * 100

    # Opsi C: 50mm
    label_50 = (df[station] > 50).astype(int)
    n_extreme_50 = label_50.sum()
    pct_50 = n_extreme_50 / len(df) * 100

    lines.append(f"\nOpsi A - Persentil-95 (threshold={p95_values[station]:.2f}mm):")
    lines.append(f"  Jumlah hari ekstrem: {n_extreme_p95} dari {len(df)} ({pct_p95:.2f}%)")

    lines.append(f"\nOpsi B - 150mm (kategori BMKG 'Hujan Sangat Lebat'):")
    lines.append(f"  Jumlah hari ekstrem: {n_extreme_150} dari {len(df)} ({pct_150:.2f}%)")

    lines.append(f"\nOpsi C - 50mm (ambang pemicu banjir, ref. Dhuha dkk. 2024):")
    lines.append(f"  Jumlah hari ekstrem: {n_extreme_50} dari {len(df)} ({pct_50:.2f}%)")

    results_summary.append({
        'stasiun': station,
        'p95_threshold': p95_values[station],
        'p95_count': n_extreme_p95, 'p95_pct': pct_p95,
        'bmkg150_count': n_extreme_150, 'bmkg150_pct': pct_150,
        'ref50_count': n_extreme_50, 'ref50_pct': pct_50
    })

# ============================================
# TABEL RINGKASAN PERBANDINGAN
# ============================================
lines.append("\n" + "="*60)
lines.append("TABEL RINGKASAN PERBANDINGAN (untuk pengambilan keputusan)")
lines.append("="*60)
summary_df = pd.DataFrame(results_summary)
lines.append(summary_df.to_string(index=False))

# ============================================
# ANALISIS: DAMPAK KE CLASS IMBALANCE
# ============================================
lines.append("\n" + "="*60)
lines.append("ANALISIS CLASS IMBALANCE (rasio kelas 0 vs 1)")
lines.append("="*60)
lines.append("\nSemakin kecil persentase 'hari ekstrem', semakin parah class imbalance-nya.")
lines.append("Class imbalance parah akan menyulitkan model belajar pola kelas minoritas (ekstrem).")
lines.append("Rule of thumb: rasio di bawah 5% sudah tergolong imbalance berat, butuh penanganan")
lines.append("khusus (SMOTE, class_weight, threshold tuning) saat pemodelan nanti.\n")

for r in results_summary:
    lines.append(f"{r['stasiun']}:")
    lines.append(f"  Persentil-95 -> rasio 1:{ (100-r['p95_pct'])/r['p95_pct']:.1f} (kelas ekstrem {r['p95_pct']:.2f}%)")
    lines.append(f"  150mm        -> rasio 1:{ (100-r['bmkg150_pct'])/r['bmkg150_pct']:.1f} (kelas ekstrem {r['bmkg150_pct']:.2f}%)" if r['bmkg150_pct'] > 0 else f"  150mm        -> TIDAK ADA kejadian sama sekali (0%)")
    lines.append(f"  50mm         -> rasio 1:{ (100-r['ref50_pct'])/r['ref50_pct']:.1f} (kelas ekstrem {r['ref50_pct']:.2f}%)" if r['ref50_pct'] > 0 else f"  50mm         -> TIDAK ADA kejadian sama sekali (0%)")

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Laporan perbandingan tersimpan di: {output_report}")
print(summary_df.to_string(index=False))