import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
data_path = 'creditcard.csv'  # Replace with your dataset path
data = pd.read_csv(data_path)

# Data Preprocessing
data['Normalized_Amount'] = data['Amount'] / data['Amount'].max()
data.drop(['Amount', 'Time'], axis=1, inplace=True)

# Exploratory Data Analysis
print("Class Distribution:")
print(data['Class'].value_counts())
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()

# Splitting the data
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Training - Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Model Evaluation
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Hyperparameter Tuning - XGBoost
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(xgb, param_grid, scoring='f1', cv=3)
grid_search.fit(X_train, y_train)

# Best Model
best_xgb = grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

# Evaluation - XGBoost
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# False Positive Reduction
thresholds = np.arange(0.1, 0.9, 0.1)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_prob = best_xgb.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_prob > threshold).astype(int)
    f1 = roc_auc_score(y_test, y_pred_custom)
    print(f"Threshold: {threshold:.1f}, AUC-ROC: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best Threshold: {best_threshold:.2f}")

# Save the model
joblib.dump(best_xgb, 'fraud_detection_model.pkl')
print("Model saved as fraud_detection_model.pkl")
