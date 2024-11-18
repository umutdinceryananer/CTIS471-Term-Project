from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd

cleaned_data_path = "cleaned_data.csv"
cleaned_data = pd.read_csv(cleaned_data_path)

X = cleaned_data.drop(columns=['normal_weight'])
y = cleaned_data['normal_weight']

from sklearn.preprocessing import LabelEncoder
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=log_reg, n_features_to_select=8)  
rfe.fit(X, y)

selected_features = X.columns[rfe.support_]

print("Selected Features by RFE:")
print(selected_features.tolist())
