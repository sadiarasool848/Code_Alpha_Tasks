import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------
# 1. Load Dataset
# ----------------------------
file_name = "unemployment_Rate_upto_11_2020.csv"
df = pd.read_csv(file_name)

# Strip extra spaces from column names
df.columns = df.columns.str.strip()

# ----------------------------
# 2. Inspect Data
# ----------------------------
print("First 5 rows:\n", df.head())
print("\nData shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# ----------------------------
# 3. Data Cleaning
# ----------------------------
# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values in unemployment column
unemployment_column = "Estimated Unemployment Rate (%)"
df[unemployment_column] = df[unemployment_column].fillna(method='ffill')

# Drop any remaining missing values
df = df.dropna(subset=['Region', 'Date', unemployment_column])

# ----------------------------
# 4. Add COVID Period Column
# ----------------------------
df['Period'] = df['Date'].apply(lambda x: 'COVID' if x.year >= 2020 else 'Pre-COVID')

# ----------------------------
# 5. Exploratory Data Analysis
# ----------------------------
print("\nSummary statistics:\n", df[unemployment_column].describe())
print("\nAverage unemployment rate by Region:\n", df.groupby('Region')[unemployment_column].mean())

# ----------------------------
# 6. Visualizations
# ----------------------------
sns.set_style("whitegrid")

# Line plot: Trend of unemployment by region
plt.figure(figsize=(14,6))
sns.lineplot(data=df, x='Date', y=unemployment_column, hue='Region', marker='o')
plt.title('Unemployment Rate Trend by Region')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxplot: Pre-COVID vs COVID
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='Period', y=unemployment_column, palette="Set2")
plt.title('Unemployment Rate: Pre-COVID vs COVID')
plt.ylabel('Unemployment Rate (%)')
plt.show()

# Histogram: Distribution of unemployment rates
plt.figure(figsize=(8,6))
sns.histplot(df[unemployment_column], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Unemployment Rates')
plt.xlabel('Unemployment Rate (%)')
plt.show()

# Bar chart: Average unemployment by region
plt.figure(figsize=(12,6))
region_avg = df.groupby('Region')[unemployment_column].mean().sort_values(ascending=False)
sns.barplot(x=region_avg.index, y=region_avg.values, palette="coolwarm")
plt.title('Average Unemployment Rate by Region')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.show()

# COVID impact per region
plt.figure(figsize=(12,6))
covid_impact = df.groupby(['Region','Period'])[unemployment_column].mean().unstack()
covid_impact.plot(kind='bar', figsize=(12,6))
plt.title('COVID Impact on Unemployment by Region')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Insights
# ----------------------------
pre_covid_mean = df[df['Period']=='Pre-COVID'][unemployment_column].mean()
covid_mean = df[df['Period']=='COVID'][unemployment_column].mean()

print(f"\nAverage Unemployment Rate Pre-COVID: {pre_covid_mean:.2f}%")
print(f"Average Unemployment Rate During COVID: {covid_mean:.2f}%")

if covid_mean > pre_covid_mean:
    print("Insight: Unemployment increased during COVID-19.")
else:
    print("Insight: Unemployment decreased or remained stable during COVID-19.")

# Top 5 regions with highest average unemployment
top_regions = df.groupby('Region')[unemployment_column].mean().sort_values(ascending=False).head(5)
print("\nTop 5 regions with highest average unemployment:\n", top_regions)
