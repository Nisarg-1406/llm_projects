import pandas as pd
import numpy as np
from scipy import stats

# Load the cleaned dataset
file_path = 'OnlineSalesData.csv'
data_cleaned = pd.read_csv(file_path)

data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])

# Calculate the Z-scores for 'Units Sold' and 'Total Revenue'
z_scores = np.abs(stats.zscore(data_cleaned[['Units Sold', 'Total Revenue']]))

# Define a threshold for identifying outliers
threshold = 3

# Identify outliers
outliers = (z_scores > threshold).any(axis=1)

# Filter the dataset to remove outliers
cleaned_data_no_outliers = data_cleaned[~outliers]

# Display the shape of the cleaned dataset without outliers
print(cleaned_data_no_outliers.shape)