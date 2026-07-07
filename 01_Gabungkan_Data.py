import pandas as pd
import glob
import os

# Karena file a1.csv sampai a11.csv ada di folder yang SAMA dengan script ini,
# kita tidak perlu navigasi kemana-mana.

# 1. Cari semua file yang namanya diawali 'a' dan berakhiran '.csv'
# Ini akan menemukan a1.csv, a2.csv, ... a11.csv
daftar_file = glob.glob("a*.csv")

# Jika ternyata nama file-nya beda (misal 2015.csv), kita bisa pakai "*.csv" saja.
# Tapi karena di screenshot namanya a1-a11, kita pakai "a*.csv".

print("=" * 50)
print("📁 File yang akan digabung:")
for f in daftar_file:
    print(f"   - {f}")

# 2. Baca dan gabungkan
list_df = []
for file in daftar_file:
    print(f"⏳ Membaca {file}...")
    df = pd.read_csv(file)
    list_df.append(df)

data_gabungan = pd.concat(list_df, ignore_index=True)

# 3. Buat folder hasil jika belum ada
os.makedirs("03_Hasil", exist_ok=True)
output_path = "03_Hasil/Data_Hujan_Gabungan_2015_2025.csv"
data_gabungan.to_csv(output_path, index=False)

print("=" * 50)
print(f"✅ Berhasil! Total baris: {len(data_gabungan)}")
print(f"💾 Disimpan di: {output_path}")