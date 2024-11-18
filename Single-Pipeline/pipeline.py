import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

file_path = 'CTIS471_2425_first_part.csv'
data = pd.read_csv(file_path)

def calculate_bmi(row):
    height_m = row['Height']
    weight_kg = row['Weight']
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)

data['Height'] = data.apply(lambda x: x['Height'] * 0.0254 if x['metric'] == 'USA' else x['Height'], axis=1)
data['Weight'] = data.apply(lambda x: x['Weight'] * 0.453592 if x['metric'] == 'USA' else x['Weight'], axis=1)
data['metric'] = 'EU'  # Metric formatını güncelle

data['BMI'] = data.apply(calculate_bmi, axis=1)

data['normal_weight'] = data['normal_weight'].apply(lambda x: 1 if x == 'yes' else 0)

data = data[(data['screen_time'] >= 0) & (data['workout'] >= 0)].copy()

small_features = ['Height', 'Weight', 'eat_refreshment', 'family_history_with_overweight', 'Age', 'BMI']
large_features = ['Height', 'Weight', 'eat_refreshment', 'family_history_with_overweight', 'Age',
                  'BMI', 'water_consumption', 'eat_junk_food', 'track_calories']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

small_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['Height', 'Weight', 'Age', 'BMI']),
        ('cat', categorical_transformer, ['eat_refreshment', 'family_history_with_overweight'])
    ])

large_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['Height', 'Weight', 'Age', 'BMI', 'water_consumption']),
        ('cat', categorical_transformer, ['eat_refreshment', 'family_history_with_overweight',
                                          'eat_junk_food', 'track_calories'])
    ])

X_small = data[small_features]
X_small_transformed = small_preprocessor.fit_transform(X_small)
small_columns = (
    ['Height', 'Weight', 'Age', 'BMI'] +
    list(small_preprocessor.transformers_[1][1].get_feature_names_out(['eat_refreshment', 'family_history_with_overweight']))
)
X_small_transformed_df = pd.DataFrame(X_small_transformed, columns=small_columns)

X_large = data[large_features]
X_large_transformed = large_preprocessor.fit_transform(X_large)
large_columns = (
    ['Height', 'Weight', 'Age', 'BMI', 'water_consumption'] +
    list(large_preprocessor.transformers_[1][1].get_feature_names_out(['eat_refreshment', 'family_history_with_overweight',
                                                                       'eat_junk_food', 'track_calories']))
)
X_large_transformed_df = pd.DataFrame(X_large_transformed, columns=large_columns)

data.to_csv('dependent_variable.csv', index=False)

X_small_transformed_df.to_csv('small_transformed.csv', index=False)
X_large_transformed_df.to_csv('large_transformed.csv', index=False)
