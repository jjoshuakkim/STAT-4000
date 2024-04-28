import pandas as pd

# Replace 'your_dataset.csv' with the actual file path
df = pd.read_csv("student+performance\student\student-mat.csv", delimiter=";")
print(df.shape)
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())

# Visualize
import matplotlib.pyplot as plt

plt.hist(df['G3'])
plt.xlabel('G3 Grades')
plt.ylabel('Frequency')
plt.show()

# Relationships
plt.scatter(df['G1'], df['G3'])
plt.xlabel('G1 Grades')
plt.ylabel('G3 Grades')
plt.show()

df = pd.get_dummies(df, columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                         'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])

# Split dataset
from sklearn.model_selection import train_test_split

X = df[['G1', 'G2']]  # Features
y = df['G3']           # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

correlation_matrix = df[['G1', 'G2', 'G3']].corr()
print(correlation_matrix)

# Correlations
correlation_matrix = df[['G1', 'G2', 'G3']].corr()
print(correlation_matrix)

# Visualize correlations
import seaborn as sns

sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('R-squared:', r2)
print('Mean Squared Error:', mse)