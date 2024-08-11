import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


file_path = '/content/advertising.csv'
df = pd.read_csv(file_path)


print(df.head())
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

duplicate_rows = df.duplicated().sum()
print("\nNumber of duplicate rows:")
print(duplicate_rows)


X = df[['TV', 'Radio', 'Newspaper']] 
y = df['Sales']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted')
plt.show()
