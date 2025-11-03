import pandas as pd
import numpy as np
# 1. Veri Yükleme
df = pd.read_csv('heart.csv')

# 2. İlk 5 Satıra Göz Atma
print("### İlk 5 Satır (df.head()) ###")
print(df.head())
print("\n" + "="*50 + "\n")

# 3. Sütun Bilgileri ve Veri Tipleri (df.info())
print("### Sütun Bilgileri ve Veri Tipleri (df.info()) ###")
df.info()
print("\n" + "="*50 + "\n")

# 4. Temel İstatistikler (df.describe())
print("### Temel İstatistikler (df.describe()) ###")
print(df.describe())

# 6. Veri Temizliği
# Eksik değer içeren satırları sil
df.dropna(inplace=True) 

# 7. Yinelenen (duplicate) satırları sil
df.drop_duplicates(inplace=True) 

numeric_cols = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak']
print("### Tüm Sayısal Sütunlarda Uç Değer Temizliği (Capping) ###")

# 8. Tüm Sayısal Sütunlara IQR Capping Uygulama
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    # Sınırlandırma işlemi
    df[col] = np.clip(df[col], lower_limit, upper_limit)
    
    print(f"✅ {col} temizlendi. Yeni Min: {df[col].min():.2f}, Yeni Max: {df[col].max():.2f}")
 
# 9. Veri Tiplerini Doğrulama (Kategorik Dönüşümler) - Temizliğin son adımı
categorical_cols = ['Sex', 'ChestPain', 'Fbs', 'RestECG', 'ExAng', 'Slope', 'Ca', 'Thal', 'Target']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print("\n" + "="*50 + "\n")
print(f"Kalan satır sayısı: {df.shape[0]}")
print("--- Son Durum ---")
print(df.dtypes)
