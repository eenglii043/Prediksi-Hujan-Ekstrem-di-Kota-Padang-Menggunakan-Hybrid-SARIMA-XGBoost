# ================================================================
# TUGAS AKHIR - SAINS DATA ITERA
# STEP 13: STATISTIK DASAR + RATA-RATA TAHUNAN PER STASIUN
# ================================================================

import pandas as pd
import numpy as np
from datetime import datetime

# 1. Baca data gabungan
df = pd.read_csv("03_Hasil/Data_Hujan_Gabungan_2015_2025.csv")

# 2. Konversi timestamp dan ekstrak tahun
df['DATA TIMESTAMP'] = pd.to_datetime(df['DATA TIMESTAMP'])
df['TAHUN'] = df['DATA TIMESTAMP'].dt.year

# 3. Filter data valid (RAINFALL DAY MM tidak NaN)
df_valid = df[df['RAINFALL DAY MM'].notna()].copy()

print("=" * 70)
print("📊 MEMBANGUN STATISTIK DASAR PER STASIUN")
print("=" * 70)
print(f"Total baris data valid: {len(df_valid)}")
print(f"Total stasiun: {df_valid['NAME'].nunique()}")
print("=" * 70)

# 4. Dapatkan daftar semua stasiun (urutkan alfabetis)
semua_stasiun = sorted(df_valid['NAME'].unique())

# ================================================================
# BUAT LAPORAN TXT
# ================================================================

def buat_laporan_statistik(output_file="03_Hasil/Statistik_Dasar_Per_Stasiun.txt"):
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("📊 STATISTIK DASAR CURAH HUJAN PER STASIUN (10 TAHUN 2015-2025)\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total stasiun: {len(semua_stasiun)}\n")
        f.write(f"Periode data: 2015 - 2025\n")
        f.write(f"Tanggal laporan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")
        
        # Loop per stasiun
        for idx, station in enumerate(semua_stasiun, 1):
            # Ambil data stasiun ini
            data_station = df_valid[df_valid['NAME'] == station]['RAINFALL DAY MM']
            
            if len(data_station) == 0:
                continue
            
            # ============================================================
            # STATISTIK DASAR (SELURUH 10 TAHUN)
            # ============================================================
            min_val = data_station.min()
            max_val = data_station.max()
            mean_val = data_station.mean()
            median_val = data_station.median()
            std_val = data_station.std()
            total_hari = len(data_station)
            total_tahun = data_station.groupby(df_valid[df_valid['NAME'] == station]['TAHUN']).ngroups
            
            f.write(f"{idx:>3}. 📍 {station}\n")
            f.write("-" * 100 + "\n")
            f.write(f"   Total Hari Data          : {total_hari:,} hari (dari {total_tahun} tahun)\n")
            f.write(f"   Minimum (mm)             : {min_val:.2f}\n")
            f.write(f"   Maximum (mm)             : {max_val:.2f}\n")
            f.write(f"   Rata-rata (Mean)         : {mean_val:.2f} mm/hari\n")
            f.write(f"   Median                   : {median_val:.2f} mm/hari\n")
            f.write(f"   Standar Deviasi          : {std_val:.2f}\n")
            
            # ============================================================
            # RATA-RATA TAHUNAN (Mean per tahun)
            # ============================================================
            annual_avg = data_station.groupby(
                df_valid[df_valid['NAME'] == station]['TAHUN']
            ).mean()
            
            f.write(f"   Rata-rata Tahunan (mm/hari):\n")
            
            # Tampilkan semua tahun 2015-2025
            for year in range(2015, 2026):
                if year in annual_avg.index:
                    val = annual_avg[year]
                    f.write(f"      - {year} : {val:>8.2f} mm/hari\n")
                else:
                    f.write(f"      - {year} : {'Data tidak tersedia':>20}\n")
            
            # Tambahkan rata-rata dari semua tahun yang tersedia
            if len(annual_avg) > 0:
                rata_rata_tahunan = annual_avg.mean()
                f.write(f"      Rata-rata seluruh tahun : {rata_rata_tahunan:.2f} mm/hari\n")
            
            f.write("\n")  # Spasi antar stasiun
        
        # ================================================================
        # RINGKASAN STATISTIK GLOBAL
        # ================================================================
        f.write("=" * 100 + "\n")
        f.write("📊 RINGKASAN STATISTIK GLOBAL (SEMUA STASIUN)\n")
        f.write("=" * 100 + "\n")
        
        # Hitung rata-rata dari semua stasiun
        all_means = df_valid.groupby('NAME')['RAINFALL DAY MM'].mean()
        all_max = df_valid.groupby('NAME')['RAINFALL DAY MM'].max()
        all_min = df_valid.groupby('NAME')['RAINFALL DAY MM'].min()
        
        f.write(f"   Rata-rata curah hujan seluruh stasiun : {all_means.mean():.2f} mm/hari\n")
        f.write(f"   Curah hujan maksimum tertinggi        : {all_max.max():.2f} mm\n")
        f.write(f"   Curah hujan minimum terendah          : {all_min.min():.2f} mm\n")
        f.write(f"   Jumlah stasiun dengan mean > 20 mm    : {(all_means > 20).sum()} stasiun\n")
        f.write(f"   Jumlah stasiun dengan mean < 5 mm     : {(all_means < 5).sum()} stasiun\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("🏆 10 STASIUN DENGAN RATA-RATA HARIAN TERTINGGI:\n")
        f.write("-" * 100 + "\n")
        top10 = all_means.sort_values(ascending=False).head(10)
        for i, (station, val) in enumerate(top10.items(), 1):
            f.write(f"   {i:>2}. {station:<35} : {val:>6.2f} mm/hari\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Laporan dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n")
    
    print(f"✅ Laporan statistik disimpan di: {output_file}")

# Jalankan
buat_laporan_statistik()