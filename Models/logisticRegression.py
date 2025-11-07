# --- 4. Model Training: Baseline (Logistic Regression) ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

print("\n--- 4. Model Training: Baseline (LogisticRegression) ---")

# Import necessary libraries for modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, auc, f1_score, precision_score, recall_score

# --- 4.1. Split Data into Features (X) and Target (y) ---

# X = all columns except the last one ('Class')
X = df.iloc[:, :-1]
# y = only the last column ('Class')
y = df.iloc[:, -1]

print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")


# --- 4.2. Create Stratified Train-Test Split ---
# We use stratify=y to ensure the test set has the same fraud ratio as the full dataset
# This is critical for imbalanced data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Fraud ratio in train set: {y_train.mean()*100:.3f}%")
print(f"Fraud ratio in test set: {y_test.mean()*100:.3f}%")
print("-" * 30)


# --- 4.3. Train Logistic Regression Model ---
print("Training Logistic Regression model...")

# Instantiate the model
# We set class_weight='balanced' as planned in the report to handle imbalance.
# 'solver' and 'random_state' are set for reproducible results.
lr_model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)

# Train the model
lr_model.fit(X_train, y_train)

print("Model training complete.")
print("-" * 30)


# --- 4.4. Evaluate Model Performance ---
print("Evaluating model performance on the test set...")

# Get class predictions (0 or 1)
y_pred = lr_model.predict(X_test)

# Get probability scores for the positive class (Fraud)
# This is needed for the AUC-PR calculation
y_scores = lr_model.predict_proba(X_test)[:, 1]

# 1. Primary Metric: AUC-PR (Average Precision Score)
auc_pr = average_precision_score(y_test, y_scores)
print(f"Area Under Precision-Recall Curve (AUC-PR): {auc_pr:.4f}")

# 2. Secondary Metrics: Precision, Recall, F1-Score (for Fraud Class)
# We can get these directly from the classification report or individual functions

# Using individual functions for clarity (labels=[1] specifies the fraud class)
fraud_precision = precision_score(y_test, y_pred, labels=[1], average='binary')
fraud_recall = recall_score(y_test, y_pred, labels=[1], average='binary')
fraud_f1 = f1_score(y_test, y_pred, labels=[1], average='binary')

print(f"Fraud Precision: {fraud_precision:.4f}")
print(f"Fraud Recall: {fraud_recall:.4f}")
print(f"Fraud F1-Score: {fraud_f1:.4f}")

# 3. Full Classification Report
print("\nFull Classification Report:")
# target_names=['Non-Fraud (0)', 'Fraud (1)'] adds clarity
print(classification_report(y_test, y_pred, target_names=['Non-Fraud (0)', 'Fraud (1)']))
print("-" * 30)


# --- 4.5. Plot Precision-Recall Curve ---
print("Plotting Precision-Recall Curve...")

# Calculate PR curve points
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Calculate the AUC (Area Under Curve) for the PR curve
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='b', label=f'LR PR Curve (AUC = {pr_auc:.4f})')
plt.title('Logistic Regression Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

print("\nBaseline model (Logistic Regression) complete.")