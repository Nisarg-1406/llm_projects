import pandas as pd

# Load the dataset
file_path = 'OnlineSalesData.csv'
data = pd.read_csv(file_path)

# Data Cleaning and Preprocessing

# Check for missing values
missing_values = data.isnull().sum()

# Drop rows with missing values (if any)
data_cleaned = data.dropna()

# Check for duplicates
duplicates = data_cleaned.duplicated().sum()

# Drop duplicates (if any)
data_cleaned = data_cleaned.drop_duplicates()

# Convert 'Date' to datetime format
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])

# Display the cleaned dataset info
print(data_cleaned.info())