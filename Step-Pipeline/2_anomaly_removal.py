import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

file_path = 'step_one.csv'
converted_data = pd.read_csv(file_path)

""""
def report_negatives(data):
    report = {}
    for column in data.select_dtypes(include=np.number).columns:
        negative_count = (data[column] < 0).sum()
        report[column] = negative_count
    return pd.DataFrame.from_dict(report, orient='index', columns=['Negative Count'])
"""

def normalize_data(data):
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=np.number).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def encode_target(data, target_column):
    # Encode the target column (yes/no -> 1/0)
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])
    return data

def clean_data(data):
    # negative_data = data[(data['screen_time'] < 0) | (data['workout'] < 0)].copy() # Created for Outlier Analysis
    cleaned_data = data[(data['screen_time'] >= 0) & (data['workout'] >= 0)].copy() 

    cleaned_data.to_csv('step_two.csv', index=False, float_format='%.5f')
    # negative_data.to_csv('removed_data.csv', index=False, float_format='%.5f') # Created for Outlier Analysis
    return cleaned_data

# print("Before cleaning:")
# print(report_negatives(converted_data))

converted_data = encode_target(converted_data, 'normal_weight')

cleaned_data = clean_data(converted_data)


# print("\nAfter cleaning:")
# print(report_negatives(cleaned_data))

# removed_rows = len(converted_data) - len(cleaned_data)
# print(f'\n{removed_rows} row removed')

""" 
plt.figure(figsize=(10, 5))
plt.hist(cleaned_data['screen_time'], bins=50, alpha=0.7, label='Screen Time')
plt.hist(cleaned_data['workout'], bins=50, alpha=0.7, label='Workout')
plt.legend()
plt.title("After Cleaning")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show() 
"""
