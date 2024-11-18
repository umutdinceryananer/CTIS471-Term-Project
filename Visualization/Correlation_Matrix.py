import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cleaned_data = pd.read_csv('CTIS471_2425_first_part.csv')

cleaned_data_numeric = cleaned_data.select_dtypes(include=['number'])

cleaned_data_encoded = pd.get_dummies(cleaned_data, drop_first=True)

corr_matrix = cleaned_data_encoded.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix for All Variables")
plt.show()
cleaned_data.head(), corr_matrix


