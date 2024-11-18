import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Random Forest uses Large Dataset
X_large_transformed_df = pd.read_csv('large_transformed.csv')

file_path = 'dependent_variable.csv' # for Single Pipeline  
# file_path = 'step_two.csv' # For Manual Pipeline  

cleaned_data = pd.read_csv(file_path)
y = cleaned_data['normal_weight'] 

large_pipeline_rf = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Random Forest model
])

# Cross-Validation
print("\nCross-Validation (Large Dataset with Random Forest):")
cv_scores_large_rf = cross_val_score(large_pipeline_rf, X_large_transformed_df, y, cv=5, scoring='accuracy')
print(f"Accuracy Scores: {cv_scores_large_rf}")
print(f"Mean Accuracy: {cv_scores_large_rf.mean():.4f} ± {cv_scores_large_rf.std():.4f}")

# Predict with Cross-Validation
y_pred_cv_large_rf = cross_val_predict(large_pipeline_rf, X_large_transformed_df, y, cv=5)
print("\nClassification Report (Large Dataset with Random Forest):")
print(classification_report(y, y_pred_cv_large_rf))

print("Confusion Matrix (Large Dataset with Random Forest):")
print(confusion_matrix(y, y_pred_cv_large_rf))
