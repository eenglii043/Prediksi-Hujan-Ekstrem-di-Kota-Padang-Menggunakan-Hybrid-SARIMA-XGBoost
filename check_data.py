import pandas as pd

df = pd.read_csv(r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Hujan_Gabungan_2015_2025.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(10))
print(df.dtypes)
print(df.isna().sum())