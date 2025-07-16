import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('OnlineSalesData.csv')

# Preprocessing the data
# # Handling missing values
df.fillna(method='ffill', inplace=True)

# Encoding categorical variables
df = pd.get_dummies(df, drop_first=True)

# Define features and target variable
X = df.drop('Sales', axis=1)  

# Assuming 'Sales' is the target variable
y = df['Sales']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Training the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
results = {'Mean Squared Error': mse, 'R^2 Score': r2}
print(results)