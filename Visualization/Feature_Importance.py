import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

cleaned_data_path = "cleaned_data.csv"
cleaned_data = pd.read_csv(cleaned_data_path)

X = cleaned_data.drop(columns=['normal_weight'])
y = cleaned_data['normal_weight']

categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le 

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Analysis')
plt.gca().invert_yaxis() 
plt.show()

print(feature_importances)
