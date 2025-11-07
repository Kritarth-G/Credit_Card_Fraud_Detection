# Import the necessary libraries
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint, uniform as sp_uniform
import numpy as np
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Re-use the settings from Random Forest
N_ITER_STAGE = 10
CV_FOLDS = 5

# --- 1. Calculate Imbalance Weight (Required for XGBoost) ---
# Assuming y_train is available globally from your previous steps
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight_value = neg_count / pos_count
print(f"Fixed scale_pos_weight: {scale_pos_weight_value:.2f}")


print("\n" + "=" * 50)
print("--- 2. XGBoost Tuning: Coarse-to-Fine Search ---")
print("=" * 50)


# --- Stage 1: Coarse Search (Broad Range) ---
print("\n--- STAGE 1: Coarse Search (10 iterations) ---")

# Define a WIDE parameter grid for XGBoost
# *** FIX: sp_uniform(loc, scale) where scale = max - loc ***
xgb_param_dist_coarse = {
    'n_estimators': sp_randint(100, 1000),
    'learning_rate': sp_uniform(0.01, 0.29),      # Range [0.01, 0.3]
    'max_depth': sp_randint(3, 15),
    'min_child_weight': sp_randint(1, 15),
    'subsample': sp_uniform(0.6, 0.4),           # Range [0.6, 1.0]
    'colsample_bytree': sp_uniform(0.6, 0.4),      # Range [0.6, 1.0]
    'reg_lambda': sp_uniform(0.1, 9.9),          # Range [0.1, 10]
}

# Instantiate the base XGBoost model with fixed parameters
# *** FIX: Removed early_stopping_rounds to prevent data leakage ***
xgb_base = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight_value, # Critical for imbalance
    eval_metric='aucpr',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

# Instantiate RandomizedSearchCV
xgb_random_search_coarse = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_dist_coarse,
    n_iter=N_ITER_STAGE,
    scoring='average_precision',
    cv=CV_FOLDS,
    verbose=2, # Set to 2 to see progress
    random_state=42,
    n_jobs=-1
)

# Execute the search
# *** FIX: Removed eval_set to prevent data leakage ***
print("Starting Coarse Search... (This may take a while)")
xgb_random_search_coarse.fit(X_train, y_train)

best_params_coarse_xgb = xgb_random_search_coarse.best_params_
print(f"Best Coarse AUC-PR: {xgb_random_search_coarse.best_score_:.4f}")
print("Coarse Parameters:")
print(best_params_coarse_xgb)


# --- Stage 2: Fine Search (Narrow Range) ---
print("\n--- STAGE 2: Fine Search (10 iterations) ---")

# Function to define a tighter range for integers
def get_fine_range_int(best_val, buffer):
    return sp_randint(max(1, best_val - buffer), best_val + buffer)

# Function to define a tighter range for floats (uniform distribution)
def get_fine_range_float(best_val, buffer, min_bound=0.001, max_bound=1.0):
    min_val = max(min_bound, best_val - buffer)
    # Use 'buffer * 2' as scale, as sp_uniform is (loc, scale)
    scale = buffer * 2
    # Adjust scale if it goes out of bounds
    if min_val + scale > max_bound:
        scale = max_bound - min_val
    return sp_uniform(min_val, scale)


# Define the NARROW parameter grid centered around best_params_coarse
# (Need to add colsample_bytree to this)
if 'colsample_bytree' not in best_params_coarse_xgb:
     best_params_coarse_xgb['colsample_bytree'] = 0.8 # Add a default if it failed

xgb_param_dist_fine = {
    'n_estimators': get_fine_range_int(best_params_coarse_xgb['n_estimators'], 200),
    'learning_rate': get_fine_range_float(best_params_coarse_xgb['learning_rate'], 0.05, max_bound=0.4),
    'max_depth': get_fine_range_int(best_params_coarse_xgb['max_depth'], 3),
    'min_child_weight': get_fine_range_int(best_params_coarse_xgb['min_child_weight'], 5),
    'subsample': get_fine_range_float(best_params_coarse_xgb['subsample'], 0.1),
    'colsample_bytree': get_fine_range_float(best_params_coarse_xgb['colsample_bytree'], 0.1),
    'reg_lambda': get_fine_range_float(best_params_coarse_xgb['reg_lambda'], 2, max_bound=20),
}

# Instantiate the fine search
xgb_random_search_fine = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_dist_fine,
    n_iter=N_ITER_STAGE,
    scoring='average_precision',
    cv=CV_FOLDS,
    verbose=2, # Set to 2 to see progress
    random_state=42,
    n_jobs=-1
)

# Execute the search
print("Starting Fine Search...")
xgb_random_search_fine.fit(X_train, y_train)

# Output Final results
print("\nXGBoost Coarse-to-Fine Tuning Complete.")
print(f"Final Best AUC-PR score: {xgb_random_search_fine.best_score_:.4f}")
print("Final Best Parameters Found:")
print(xgb_random_search_fine.best_params_)
print("-" * 50)

# Store the best model and score
best_xgb_model_final = xgb_random_search_fine.best_estimator_

# --- 6.4. Evaluate Final Tuned Model on Test Set ---
print("\n--- 6.4. Evaluating Final Tuned Model ---")

# Get class predictions and probability scores using the TUNED model
y_pred_xgb_final = best_xgb_model_final.predict(X_test)
y_scores_xgb_final = best_xgb_model_final.predict_proba(X_test)[:, 1]

# 1. Primary Metric: AUC-PR
auc_pr_xgb_final = average_precision_score(y_test, y_scores_xgb_final)
print(f"Area Under Precision-Recall Curve (AUC-PR on Test Set): {auc_pr_xgb_final:.4f}")

# 2. Secondary Metrics: Precision, Recall, F1-Score (for Fraud Class)
fraud_precision_xgb_final = precision_score(y_test, y_pred_xgb_final, labels=[1], average='binary')
fraud_recall_xgb_final = recall_score(y_test, y_pred_xgb_final, labels=[1], average='binary')
fraud_f1_xgb_final = f1_score(y_test, y_pred_xgb_final, labels=[1], average='binary')

print(f"Fraud Precision: {fraud_precision_xgb_final:.4f}")
print(f"Fraud Recall: {fraud_recall_xgb_final:.4f}")
print(f"Fraud F1-Score: {fraud_f1_xgb_final:.4f}")

# 3. Full Classification Report
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred_xgb_final, target_names=['Non-Fraud (0)', 'Fraud (1)']))
print("-" * 30)


# --- 6.5. Plot Comparative Precision-Recall Curve (using final tuned score) ---

# Calculate PR curve points for the Final Tuned XGBoost
precision_xgb_final, recall_xgb_final, _ = precision_recall_curve(y_test, y_scores_xgb_final)
pr_auc_xgb_final = auc(recall_xgb_final, precision_xgb_final)

plt.figure(figsize=(12, 8))

# Plot the baseline Logistic Regression curve (assuming variables are available)
plt.plot(recall, precision, color='blue', lw=2, linestyle=':',
         label=f'Logistic Regression (AUC-PR = {pr_auc:.4f})')

# Plot the intermediate Random Forest curve (assuming variables are available)
plt.plot(recall_rf, precision_rf, color='green', lw=2, linestyle='--',
         label=f'Random Forest (AUC-PR = {pr_auc_rf:.4f})')

# Plot the Final Tuned Advanced XGBoost curve
plt.plot(recall_xgb_final, precision_xgb_final, color='red', lw=4,
         label=f'TUNED XGBoost (AUC-PR = {pr_auc_xgb_final:.4f})')

plt.title('Final Precision-Recall Curve Comparison (Tuned Models)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

print("\nFinal Tuned Advanced model (XGBoost) successfully evaluated.")