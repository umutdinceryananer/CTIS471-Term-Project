import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

X_large_transformed_df = pd.read_csv('large_transformed.csv')

file_path = 'dependent_variable.csv' # for Single Pipeline  
# file_path = 'step_two.csv' # For Manual Pipeline  
cleaned_data = pd.read_csv(file_path)
y = cleaned_data['normal_weight']  

large_pipeline = Pipeline([
    ('classifier', LogisticRegression(max_iter=1000))  
])

print("\nCross-Validation (Large Dataset):")
cv_scores_large = cross_val_score(large_pipeline, X_large_transformed_df, y, cv=5, scoring='accuracy')
print(f"Accuracy Scores: {cv_scores_large}")
print(f"Mean Accuracy: {cv_scores_large.mean():.4f} ± {cv_scores_large.std():.4f}")

y_pred_cv_large = cross_val_predict(large_pipeline, X_large_transformed_df, y, cv=5)
print("\nClassification Report (Large Dataset):")
print(classification_report(y, y_pred_cv_large))

print("Confusion Matrix (Large Dataset):")
print(confusion_matrix(y, y_pred_cv_large))
