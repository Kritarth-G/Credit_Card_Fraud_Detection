# --- 5. Model Tuning: Intermediate (Random Forest) ---
print("\n--- 5. Model Tuning: Intermediate (RandomForest) ---")

# Import the new libraries for tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# --- 5.1. Define Parameter Grid for Tuning ---
print("Defining hyperparameter search grid...")

# Define the grid of parameters to search.
# These are distributions for RandomizedSearch to sample from.
param_dist = {
    'n_estimators': [100, 200, 500],       # Number of trees
    'max_depth': [10, 20, 30, None],       # Max depth (None=unlimited)
    'max_features': ['sqrt', 'log2'],      # Num features to consider at each split
    'min_samples_split': [2, 5, 10],       # Min samples to split a node
    'min_samples_leaf': [1, 2, 4],         # Min samples at a leaf node
    'bootstrap': [True, False]             # Whether to bootstrap samples
}

# Create a base Random Forest model to tune
# We set the parameters that we are NOT tuning
rf_base = RandomForestClassifier(
    class_weight='balanced',  # Still critical for imbalance
    n_jobs=1,                # Use all available CPUs
    random_state=42
)

# --- 5.2. Set Up Randomized Search CV ---
print("Setting up Randomized Search for Random Forest...")

# Instantiate the RandomizedSearchCV object
# n_iter = number of random combinations to try (e.g., 20). More is better but slower.
# cv = 3 (3-fold cross-validation is a fast and solid choice)
# scoring = 'average_precision' (This is the scikit-learn string for AUC-PR)
# verbose = 2 (This will print updates so you can see its progress)
rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=10,  # Try 10 different combinations (you can increase this)
    cv=2,       # Use 2-fold stratified cross-validation
    scoring='average_precision', # IMPORTANT: Optimize for AUC-PR
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# --- 5.3. Run the Hyperparameter Search ---
print("Running hyperparameter search... This may take several minutes.")
# This command runs the search on the *training data only*
rf_search.fit(X_train, y_train)

print("Search complete.")
print("-" * 30)

# --- 5.4. Get Best Model and Evaluate ---
print(f"Best parameters found: {rf_search.best_params_}")
print(f"Best AUC-PR score during search: {rf_search.best_score_:.4f}")
print("-" * 30)

# Get the best model found by the search
# This is the new, tuned model
best_rf_model = rf_search.best_estimator_

# --- 5.5. Evaluate Best Model on the *Test Set* ---
print("Evaluating the *best* model on the hold-out test set...")

# Get class predictions (0 or 1)
y_pred_rf = best_rf_model.predict(X_test)

# Get probability scores for the positive class (Fraud)
y_scores_rf = best_rf_model.predict_proba(X_test)[:, 1]

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
print("\nFull Classification Report (Tuned Model):")
print(classification_report(y_test, y_pred_rf, target_names=['Non-Fraud (0)', 'Fraud (1)']))
print("-" * 30)


# --- 5.6. Plot Comparative Precision-Recall Curve ---
print("Plotting Precision-Recall Curve (Tuned RF vs. LR)...")

# Calculate PR curve points for the new Tuned Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_scores_rf)
pr_auc_rf = auc(recall_rf, precision_rf)

# Create the plot
plt.figure(figsize=(10, 7))

# Plot the baseline Logistic Regression curve (variables are from the previous code block)
plt.plot(recall, precision, color='blue', lw=2,
         label=f'Logistic Regression (AUC-PR = {pr_auc:.4f})')

# Plot the new Tuned Random Forest curve
plt.plot(recall_rf, precision_rf, color='green', lw=2,
         label=f'Tuned Random Forest (AUC-PR = {pr_auc_rf:.4f})')

plt.title('Precision-Recall Curve Comparison')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

print("\nIntermediate model (Tuned Random Forest) complete.")