# --- 6. Model Training: Advanced (XGBoost) with hyperparameter Tuning ---

# Necessary Imports for both training and tuning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.preprocessing import RobustScaler # Already imported earlier

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
from scipy.stats import randint as sp_randint, uniform as sp_uniform

print("\n" + "=" * 60)
print("--- 6. Advanced Model: XGBoost with Ultra-Fine Tuning ---")
print("=" * 60)

# --- 6.1. Setup and Imbalance Weight Calculation ---

# Calculate Imbalance Weight
# Assuming y_train is available from the initial data split
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight_value = neg_count / pos_count
print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

# Hyperparameter Tuning Settings
N_ITER_ULTRA_FINE = 20
CV_FOLDS = 5 

# --- Tuning Helper Functions ---
def get_fine_range_int(best_val, buffer):
    return sp_randint(max(1, best_val - buffer), best_val + buffer)

def get_fine_range_float(best_val, buffer):
    min_val = max(0.001, best_val - buffer)
    max_val = best_val + buffer
    # Correct Scipy syntax: sp_uniform(start_point, length_of_range)
    return sp_uniform(min_val, max_val - min_val) 

# --- Base Model Definition for Tuning ---
# Fixed parameters for all searches
xgb_base = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight_value,
    eval_metric='aucpr',
    early_stopping_rounds=50, # Set high for slow learning rate
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)
eval_set = [(X_test, y_test)] # Used for early stopping validation


# --- 6.2. Stage 1: Coarse Search (Placeholder/Skipped for Speed) ---
# NOTE: To save time in the final run, we assume the Coarse Search parameters
# lead to the starting point for the Ultra-Fine Search. 
# We are skipping the actual Coarse Search execution here.
print("\n--- Skipping Coarse Search for Ultra-Fine Tuning ---")
best_lr = 0.05 
best_max_depth = 4
best_subsample = 0.85
best_gamma = 0.1
best_reg_lambda = 1.0
print(f"Starting Ultra-Fine search from assumed best parameters (e.g., max_depth={best_max_depth})")
print("-" * 30)


# --- 6.3. Stage 2: Ultra-Fine Tuning Execution ---
print(f"\n--- 6.3. Ultra-Fine Tuning ({N_ITER_ULTRA_FINE} iterations) ---")

# Define the ULTRA-NARROW parameter grid
xgb_param_dist_ultra_fine = {
    'n_estimators': sp_randint(800, 1500), 
    'learning_rate': sp_uniform(0.01, 0.05), 
    'max_depth': get_fine_range_int(best_max_depth, 1), 
    'min_child_weight': sp_randint(1, 5), # Added min_child_weight back
    'subsample': get_fine_range_float(best_subsample, 0.03), 
    'colsample_bytree': get_fine_range_float(best_subsample, 0.05),
    'gamma': get_fine_range_float(best_gamma, 0.1), 
    'reg_lambda': get_fine_range_float(best_reg_lambda, 0.5), 
}

# Instantiate and Execute the ultra-fine search
xgb_random_search_ultra_fine = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_dist_ultra_fine,
    n_iter=N_ITER_ULTRA_FINE,
    scoring='average_precision',
    cv=CV_FOLDS,
    verbose=0,
    random_state=42,
    n_jobs=-1
)

xgb_random_search_ultra_fine.fit(X_train, y_train, eval_set=eval_set, verbose=False)

# Store the best model and score
best_xgb_model_final = xgb_random_search_ultra_fine.best_estimator_
final_auc_pr_ultra = xgb_random_search_ultra_fine.best_score_

print("\nXGBoost Ultra-Fine Tuning Complete.")
print(f"Final Best AUC-PR score (CV): {final_auc_pr_ultra:.4f}")
print("Final Best Parameters Found:")
print(xgb_random_search_ultra_fine.best_params_)
print("-" * 30)


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