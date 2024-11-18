import pandas as pd

file_path = 'CTIS471_2425_first_part.csv'

data = pd.read_csv(file_path)

def calculate_bmi(row):
    height_m = row['Height'] 
    weight_kg = row['Weight'] 
    bmi = weight_kg / (height_m ** 2)  
    return round(bmi, 1)  

converted_data = data.copy()

us_indices = converted_data['metric'] == 'USA'

converted_data.loc[us_indices, 'Height'] = converted_data.loc[us_indices, 'Height'] * 0.0254

converted_data.loc[us_indices, 'Weight'] = converted_data.loc[us_indices, 'Weight'] * 0.453592

converted_data.loc[us_indices, 'metric'] = 'EU'

converted_data['BMI'] = converted_data.apply(calculate_bmi, axis=1)

output_file = 'step_one.csv'
converted_data.to_csv(output_file, index=False, float_format='%.5f')
