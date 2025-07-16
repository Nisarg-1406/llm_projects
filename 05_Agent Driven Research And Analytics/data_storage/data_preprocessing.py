import pandas as pd
import numpy as np

# Load the dataset
file_path = 'OnlineSalesData.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print("Initial Data:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values by filling with mean for numeric columns
# (if there were any missing values)
for column in df.select_dtypes(include=[np.number]).columns:
    df[column].fillna(df[column].mean(), inplace=True)

# Check for duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    df.drop_duplicates(inplace=True)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check for outliers using IQR method
Q1 = df['Total Revenue'].quantile(0.25)
Q3 = df['Total Revenue'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Total Revenue'] < lower_bound) | (df['Total Revenue'] > upper_bound)]

print("\nOutliers in Total Revenue:")
print(outliers)