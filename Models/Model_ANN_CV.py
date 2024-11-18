import tensorflow as tf
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from scikeras.wrappers import KerasClassifier
import pandas as pd

# ANN uses Large Dataset
X_large_transformed_df = pd.read_csv('./large_transformed.csv')

file_path = 'dependent_variable.csv' # for Single Pipeline  
# file_path = 'step_two.csv' # For Manual Pipeline  

cleaned_data = pd.read_csv(file_path)
y = cleaned_data['normal_weight'] 

def create_ann(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

print("\nCross-Validation (Large Dataset with ANN):")
ann_model_large = KerasClassifier(model=create_ann, input_dim=X_large_transformed_df.shape[1], epochs=50, batch_size=16, verbose=0)
cv_scores_ann_large = cross_val_score(ann_model_large, X_large_transformed_df, y, cv=5)
print(f"Accuracy Scores (Large Dataset): {cv_scores_ann_large}")
print(f"Mean Accuracy: {cv_scores_ann_large.mean():.4f} ± {cv_scores_ann_large.std():.4f}")

y_pred_ann_large = cross_val_predict(ann_model_large, X_large_transformed_df, y, cv=5)
print("\nClassification Report (Large Dataset):")
print(classification_report(y, y_pred_ann_large))
print("Confusion Matrix (Large Dataset):")
print(confusion_matrix(y, y_pred_ann_large))
