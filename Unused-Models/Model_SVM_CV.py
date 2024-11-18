from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

X_small_transformed_df = pd.read_csv('small_transformed.csv')

file_path = 'dependent_variable.csv' # for Single Pipeline  
# file_path = 'step_two.csv' # For Manual Pipeline  
cleaned_data = pd.read_csv(file_path)
y = cleaned_data['normal_weight']

svm_model_small = SVC(kernel='rbf', class_weight='balanced', random_state=42)

print("Cross-Validation (Small Dataset with SVM):")
cv_scores_svm_small = cross_val_score(svm_model_small, X_small_transformed_df, y, cv=5, scoring='accuracy')
print(f"Accuracy Scores (Small Dataset): {cv_scores_svm_small}")
print(f"Mean Accuracy: {cv_scores_svm_small.mean():.4f} ± {cv_scores_svm_small.std():.4f}")

y_pred_svm_small = cross_val_predict(svm_model_small, X_small_transformed_df, y, cv=5)
print("\nClassification Report (Small Dataset):")
print(classification_report(y, y_pred_svm_small))
print("Confusion Matrix (Small Dataset):")
print(confusion_matrix(y, y_pred_svm_small))