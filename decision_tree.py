import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

# ===== Save outputs to a file =====
sys.stdout = open("model_results.txt", "w")

# ===== Load Dataset =====
df = pd.read_csv("D:\TASKS\TASK5\heart.csv")
X = df.drop('target', axis=1)
y = df['target']

print("Dataset Shape:", df.shape)
print("First 5 Rows:\n", df.head())

# ===== Train-Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== Decision Tree Hyperparameter Tuning =====
dt_params = {
    'max_depth': [3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params, cv=5, scoring='accuracy', n_jobs=-1
)
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_

print("\nBest Decision Tree Params:", dt_grid.best_params_)
y_pred_dt = best_dt.predict(X_test)
print("\nOptimized Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))

# ===== Random Forest Hyperparameter Tuning =====
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params, cv=5, scoring='accuracy', n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print("\nBest Random Forest Params:", rf_grid.best_params_)
y_pred_rf = best_rf.predict(X_test)
print("\nOptimized Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# ===== Confusion Matrix Function =====
def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=['No Disease','Disease'],
                yticklabels=['No Disease','Disease'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

save_confusion_matrix(y_test, y_pred_dt, "Optimized Decision Tree", "confusion_matrix_dt.png")
save_confusion_matrix(y_test, y_pred_rf, "Optimized Random Forest", "confusion_matrix_rf.png")

# ===== Cross-validation scores =====
cv_dt = cross_val_score(best_dt, X, y, cv=5).mean()
cv_rf = cross_val_score(best_rf, X, y, cv=5).mean()

print("\nMean CV Accuracy (Optimized Decision Tree):", round(cv_dt, 4))
print("Mean CV Accuracy (Optimized Random Forest):", round(cv_rf, 4))

# ===== Final Comparison Table =====
comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Test Accuracy": [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf)],
    "CV Accuracy": [cv_dt, cv_rf]
})
print("\nFinal Model Comparison:\n", comparison)

# Restore stdout
sys.stdout.close()
sys.stdout = sys.__stdout__

print(" All outputs saved to 'model_results.txt'")
print(" Confusion matrices saved as PNGs.")
