import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'OnlineSalesData.csv'
df = pd.read_csv(file_path)

# Convert 'Date' to datetime

df['Date'] = pd.to_datetime(df['Date'])

# Set the style of seaborn
sns.set(style='whitegrid')

# 1. Distribution of Total Revenue
plt.figure(figsize=(10, 6))
sns.histplot(df['Total Revenue'], bins=30, kde=True)
plt.title('Distribution of Total Revenue')
plt.xlabel('Total Revenue')
plt.ylabel('Frequency')
plt.savefig('total_revenue_distribution.png')
plt.close()

# 2. Correlation Heatmap
# Selecting only numerical columns for correlation
numerical_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
correlation = numerical_df.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# 3. Sales trends over time
plt.figure(figsize=(14, 7))
sales_trend = df.groupby('Date')['Total Revenue'].sum().reset_index()
sns.lineplot(data=sales_trend, x='Date', y='Total Revenue', marker='o')
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sales_trends_over_time.png')
plt.close()

# 4. Distribution of sales by product category
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Product Category', order=df['Product Category'].value_counts().index)
plt.title('Distribution of Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Number of Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sales_distribution_by_category.png')
plt.close()

# 5. Total revenue by region
plt.figure(figsize=(12, 6))
total_revenue_region = df.groupby('Region')['Total Revenue'].sum().reset_index()
sns.barplot(data=total_revenue_region, x='Region', y='Total Revenue', palette='viridis')
plt.title('Total Revenue by Region')
plt.xlabel('Region')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('total_revenue_by_region.png')
plt.close()