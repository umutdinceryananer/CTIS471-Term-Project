import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X_small_transformed_df = pd.read_csv('small_transformed.csv')

file_path = 'dependent_variable.csv' # for Single Pipeline  
# file_path = 'step_two.csv' # For Manual Pipeline  
cleaned_data = pd.read_csv(file_path)
y = cleaned_data['normal_weight']  

small_pipeline_knn = Pipeline([
    ('classifier', KNeighborsClassifier(n_neighbors=3))  # KNN model with 3 neighbors
])

print("Cross-Validation (Small Dataset with KNN):")
cv_scores_small_knn = cross_val_score(small_pipeline_knn, X_small_transformed_df, y, cv=5, scoring='accuracy')
print(f"Accuracy Scores: {cv_scores_small_knn}")
print(f"Mean Accuracy: {cv_scores_small_knn.mean():.4f} ± {cv_scores_small_knn.std():.4f}")

y_pred_cv_small_knn = cross_val_predict(small_pipeline_knn, X_small_transformed_df, y, cv=5)
print("\nClassification Report (Small Dataset with KNN):")
print(classification_report(y, y_pred_cv_small_knn))

print("Confusion Matrix (Small Dataset with KNN):")
print(confusion_matrix(y, y_pred_cv_small_knn))

""" 
param_grid = {'classifier__n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(small_pipeline_knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_small_transformed_df, y)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_) 
"""
