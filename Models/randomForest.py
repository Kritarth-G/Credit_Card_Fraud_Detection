# Random Forest with Hyperparameter Tuning
print("\n--- 5. Model Training: Intermediate (RandomForest) ---")

# Import the new model
from sklearn.ensemble import RandomForestClassifier

# --- 5.1. Train Random Forest Model ---
print("Training Random Forest model...")

# Instantiate the model
# We use the parameters discussed in the report:
# class_weight='balanced' handles imbalance.
# n_estimators and max_depth are key tuning parameters.
# n_jobs=-1 uses all CPU cores for faster training.
rf_model = RandomForestClassifier(
    n_estimators=100,         # Number of trees (tunable)
    max_depth=20,             # Max depth of trees (tunable)
    class_weight='balanced',  # Handles imbalance
    n_jobs=-1,                # Use all available CPUs
    random_state=42
)

# Train the model (using the same data as before)
rf_model.fit(X_train, y_train)

print("Model training complete.")
print("-" * 30)


# --- 5.2. Evaluate Model Performance ---
print("Evaluating model performance on the test set...")

# Get class predictions (0 or 1)
y_pred_rf = rf_model.predict(X_test)

# Get probability scores for the positive class (Fraud)
y_scores_rf = rf_model.predict_proba(X_test)[:, 1]

# 1. Primary Metric: AUC-PR (Average Precision Score)
auc_pr_rf = average_precision_score(y_test, y_scores_rf)
print(f"Area Under Precision-Recall Curve (AUC-PR): {auc_pr_rf:.4f}")

# 2. Secondary Metrics: Precision, Recall, F1-Score (for Fraud Class)
fraud_precision_rf = precision_score(y_test, y_pred_rf, labels=[1], average='binary')
fraud_recall_rf = recall_score(y_test, y_pred_rf, labels=[1], average='binary')
fraud_f1_rf = f1_score(y_test, y_pred_rf, labels=[1], average='binary')

print(f"Fraud Precision: {fraud_precision_rf:.4f}")
print(f"Fraud Recall: {fraud_recall_rf:.4f}")
print(f"Fraud F1-Score: {fraud_f1_rf:.4f}")

# 3. Full Classification Report
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Non-Fraud (0)', 'Fraud (1)']))
print("-" * 30)


# --- 5.3. Plot Comparative Precision-Recall Curve ---
print("Plotting Precision-Recall Curve (RF vs. LR)...")

# Calculate PR curve points for Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_scores_rf)
pr_auc_rf = auc(recall_rf, precision_rf)

# Create the plot
plt.figure(figsize=(10, 7))

# Plot the baseline Logistic Regression curve (variables are from the previous code block)
plt.plot(recall, precision, color='blue', lw=2,
         label=f'Logistic Regression (AUC-PR = {pr_auc:.4f})')

# Plot the new Random Forest curve
plt.plot(recall_rf, precision_rf, color='green', lw=2,
         label=f'Random Forest (AUC-PR = {pr_auc_rf:.4f})')

plt.title('Precision-Recall Curve Comparison')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

print("\nIntermediate model (Random Forest) complete.")