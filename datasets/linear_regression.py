import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('student+performance\student\student-mat.csv', delimiter=';')

# Select relevant features
features = ['age', 'absences', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'G1', 'G2']

# Preprocess the data (handle missing values, encode categorical variables if needed)

# Define features and target variable
X = df[features]
y = df['G3']

# Initialize and fit the multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Interpret the coefficients
coefficients = pd.DataFrame({'Variable': features, 'Coefficient': model.coef_})
print(coefficients)

# Evaluate the model
print('R-squared:', model.score(X, y))
predictions = model.predict(X)
plt.scatter(y, predictions)
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Actual vs. Predicted G3')
plt.show()

sns.pairplot(df[features])
plt.show()