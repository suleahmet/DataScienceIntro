import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    print(f"✅ {col} cleaned. New Min: {df[col].min():.2f}, New Max: {df[col].max():.2f}")
 
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

# Grafiklerin çizileceği sütunlar
categorical_for_plot = ['Sex', 'Target'] 

# Grafik başlıkları için daha açıklayıcı etiketler
plot_titles = {
    'Sex': 'Cinsiyet Dağılımı (0: Kadın, 1: Erkek)',
    'Target': 'Kalp Hastalığı Durumu (0: Sağlıklı, 1: Hasta)'
}

# 1 satır ve 2 sütundan oluşan bir figür oluştur
plt.figure(figsize=(12, 5))
plt.suptitle('Kategorik Özelliklerin Dağılımı (Target ve Sex)', fontsize=16, y=1.05)

for i, col in enumerate(categorical_for_plot):
    plt.subplot(1, 2, i + 1)
    
    # sns.countplot() kullanarak bar grafiği çiz
    sns.countplot(x=col, data=df, palette='viridis', edgecolor='black')
    
    # Grafiğe başlık ve etiket ekle
    plt.title(plot_titles[col], fontsize=12)
    plt.xlabel(col)
    plt.ylabel('Hasta Sayısı (Count)')

# Grafiklerin birbirine karışmamasını sağla
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('categorical_bar_charts.png')

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

# List of numerical columns for univariate analysis
numerical_cols = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak']

# ==============================================================================
# 1. Univariate Analysis (Numerical Distributions)
#    (Histograms to show the distribution shape and density)
# ==============================================================================
print("--- 1. Generating Numerical Distribution Histograms ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten() # Flatten the 2x3 grid into a 1D array for easy iteration

for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, bins=15, ax=axes[i], color='teal')
    axes[i].set_title(f'Distribution of {col}', fontsize=14)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Remove the unused 6th subplot
fig.delaxes(axes[5])
plt.suptitle('Univariate Analysis: Distribution of Key Numerical Features', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('numerical_histograms.png')
plt.close(fig)
print("✅ numerical_histograms.png saved.")


# ==============================================================================
# 2. Bivariate Analysis (Relationships)
# ==============================================================================
print("--- 2. Generating Bivariate Analysis Plots ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Bivariate Analysis: Feature Relationships', fontsize=18)

# A) Relationship between Age and Cholesterol, colored by Target
# Look for clusters: are older patients with high cholesterol more likely to be target=1?
sns.scatterplot(
    x='Age',
    y='Chol',
    hue='Target',
    data=df,
    ax=axes[0],
    palette='viridis',
    s=70, # size of points
    alpha=0.7
)

axes[0].set_title('Age vs. Cholesterol (Colored by Heart Disease)', fontsize=14)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Cholesterol')
axes[0].legend(title='Heart Disease', labels=['No (0)', 'Yes (1)'])

# B) Resting Blood Pressure (RestBP) difference between Males and Females
# Use a box plot to compare distribution characteristics (median, spread)
sns.boxplot(
    x='Sex',
    y='RestBP',
    data=df,
    ax=axes[1],
    palette='Set2'
)
axes[1].set_title('Resting Blood Pressure by Gender', fontsize=14)
axes[1].set_xlabel('Gender (0=Female, 1=Male)')
axes[1].set_ylabel('Resting Blood Pressure (RestBP)')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('bivariate_analysis.png')
plt.close(fig)
print("✅ bivariate_analysis.png saved.")


# ==============================================================================
# 3. Correlation Analysis (Heatmap)
# ==============================================================================
print("--- 3. Generating Correlation Heatmap ---")

# Calculate correlation matrix. Convert categorical columns back to their numeric codes
# for correlation calculation, as 'category' type cannot be correlated directly.
# We will use the original numeric columns list + the categorical columns that are
# still stored as numerical representations (0, 1, 2, 3, etc.)
corr_df = df.copy()
# Ensure all columns are numeric for correlation calculation
for col in corr_df.select_dtypes(include='category').columns:
    corr_df[col] = corr_df[col].cat.codes

correlation_matrix = corr_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f", # Display two decimal places
    linewidths=.5,
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()
print("✅ correlation_heatmap.png saved.")
print("\nEDA Visualization code execution complete.")