import pandas as pd
import numpy as np
# 1. Data Loading
df = pd.read_csv('heart.csv')

# 2. First 5 Rows Inspection
print("### First 5 Rows (df.head()) ###")
print(df.head())
print("\n" + "="*50 + "\n")

# 3. Column Information and Data Types (df.info())
print("### Column Information and Data Types (df.info()) ###")
df.info()
print("\n" + "="*50 + "\n")

# 4. Basic Statistics (df.describe())
print("### Basic Statistics (df.describe()) ###")
print(df.describe())

# 6. Data Cleaning
# Drop rows with missing values
df.dropna(inplace=True) 

# 7. Remove duplicate rows
df.drop_duplicates(inplace=True) 

numeric_cols = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak']
print("### Outlier Cleaning (Capping) on All Numerical Columns ###")

# 8. Apply IQR Capping to All Numerical Columns
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    # Capping operation
    df[col] = np.clip(df[col], lower_limit, upper_limit)
    
    print(f" {col} cleaned. New Min: {df[col].min():.2f}, New Max: {df[col].max():.2f}")
 
# 9. Data Type Verification (Categorical Conversions) - Final step of cleaning
# Converting columns that should be categorical/nominal to 'category' type
categorical_cols = ['Sex', 'ChestPain', 'Fbs', 'RestECG', 'ExAng', 'Slope', 'Ca', 'Thal', 'Target']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print("\n" + "="*50 + "\n")
print(f"Remaining row count: {df.shape[0]}")
print("--- Final Status ---")
print(df.dtypes)
