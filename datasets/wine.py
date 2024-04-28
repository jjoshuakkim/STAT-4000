import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("student+performance\student\student-mat.csv", delimiter=";")

# Select quantitative columns
quantitative_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                       'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 
                       'absences', 'G1', 'G2', 'G3']

# Create a new DataFrame with only quantitative data
quantitative_df = df[quantitative_columns]

# Generate scatter plots for each pair of quantitative variables
pd.plotting.scatter_matrix(quantitative_df, alpha=0.2, figsize=(15, 15), diagonal='kde')
plt.show()