import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

file_path = 'step_two.csv'
cleaned_data = pd.read_csv(file_path)

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

X_small = cleaned_data[small_features]
X_small_transformed = small_preprocessor.fit_transform(X_small)

# Large dataset
X_large = cleaned_data[large_features]
X_large_transformed = large_preprocessor.fit_transform(X_large)

small_columns = (
    ['Height', 'Weight', 'Age', 'BMI'] + 
    list(small_preprocessor.transformers_[1][1].get_feature_names_out(['eat_refreshment', 'family_history_with_overweight']))
)
large_columns = (
    ['Height', 'Weight', 'Age', 'BMI', 'water_consumption'] + 
    list(large_preprocessor.transformers_[1][1].get_feature_names_out(['eat_refreshment', 'family_history_with_overweight', 
                                                                       'eat_junk_food', 'track_calories']))
)

X_small_transformed_df = pd.DataFrame(X_small_transformed, columns=small_columns)
X_large_transformed_df = pd.DataFrame(X_large_transformed, columns=large_columns)

X_small_transformed_df.to_csv('small_transformed.csv', index=False)
X_large_transformed_df.to_csv('large_transformed.csv', index=False)

