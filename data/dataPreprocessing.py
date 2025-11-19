import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# Set visualization style
sns.set(style="whitegrid")

# --- 1. Load Dataset ---
try:
    # Load the dataset
    df = pd.read_csv('data/creditcard.csv')
    print("Dataset loaded successfully.")
    print(f"Original shape of the data: {df.shape}")
    print("-" * 30)
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found.")
    print("Please make sure the dataset is in the same directory as the script.")
    exit() # Stop the script if the file isn't found

# --- 2. Exploratory Data Analysis (EDA) ---
print("Starting Exploratory Data Analysis...")

# --- 2.1. Check for Missing Values ---
print("\n--- 2.1. Check for Missing Values ---")
total_missing = df.isnull().sum().sum()
print(f"Total missing values found: {total_missing}")

if total_missing > 0:
    # Strategy: Drop rows with missing values.
    print(f"\nDropping {total_missing} rows containing missing values...")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True) # Reset index after dropping
    print(f"Dataset shape after dropping NaNs: {df.shape}")
else:
    print("No missing values found. Dataset is clean.")
print("-" * 30)


# --- 2.2. Analyze Raw Feature Distributions (Justification for Scaling) ---
print("\n--- 2.2. Analyze Raw Feature Distributions (Justification for Scaling) ---")
print("Plotting original 'Time' and 'Amount' distributions...")

# Show visualizations
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))

# Plot for 'Amount'
sns.histplot(df['Amount'], bins=100, ax=ax1, color='b', kde=True)
ax1.set_title('Distribution of Transaction Amount (Original)')
ax1.set_xlabel('Amount ($)')
ax1.set_ylabel('Frequency')
ax1.set_xlim(0, 3000) # Limit x-axis to see the main distribution
ax1.text(0.5, 0.9, f"Max value: {df['Amount'].max():.2f}",
         transform=ax1.transAxes, ha="center")

# Plot for 'Time'
sns.histplot(df['Time'], bins=100, ax=ax2, color='r', kde=True)
ax2.set_title('Distribution of Transaction Time (Original)')
ax2.set_xlabel('Time (in seconds)')
ax2.set_ylabel('Frequency')

plt.suptitle('Original Distributions', fontsize=16)
plt.show()

print("==> INSIGHT: 'Time' and 'Amount' are heavily skewed and on different scales than V features. Scaling is required.")
print("-" * 30)

# --- 2.3. Analyze Behavior by Class ---
print("\n--- 2.3. Analyze Behavior by Class ---")
# This analysis still uses the *original* 'Time' and 'Amount'
# to keep the x-axis interpretable (e.g., in dollars)

print("\nDescriptive statistics for Fraudulent 'Amount':")
print(df[df['Class'] == 1]['Amount'].describe())
print("\nDescriptive statistics for Non-Fraudulent 'Amount':")
print(df[df['Class'] == 0]['Amount'].describe())

# Plotting transaction amount distribution
plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['Class'] == 0]['Amount'],
            label='Non-Fraud (Class 0)', fill=True, alpha=0.4)
sns.kdeplot(df[df['Class'] == 1]['Amount'],
            label='Fraud (Class 1)', fill=True, alpha=0.7)
plt.title('Transaction Amount Distribution by Class')
plt.xlabel('Amount ($)')
plt.ylabel('Density')
plt.xlim(0, 500) # Zoom in on smaller amounts
plt.legend()
plt.show()


# Time distribution for fraud vs. non-fraud
plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['Class'] == 0]['Time'],
            label='Non-Fraud (Class 0)', fill=True, alpha=0.4)
sns.kdeplot(df[df['Class'] == 1]['Time'],
            label='Fraud (Class 1)', fill=True, alpha=0.7)
plt.title('Transaction Time Distribution by Class')
plt.xlabel('Time (in seconds)')
plt.ylabel('Density')
plt.xlim(left=0)
plt.legend()
plt.show()
print("==> INSIGHT: Fraudulent transactions have a different distribution for 'Amount' and 'Time' than non-fraudulent ones.")
print("-" * 30)


# --- 2.4. Analyze Class Imbalance ---
print("\n--- 2.4. Analyze Class Imbalance ---")
class_counts = df['Class'].value_counts()
fraud_percentage = class_counts[1] / class_counts.sum() * 100

print(f"Total transactions: {len(df)}")
print(f"Fraudulent transactions (Class 1): {class_counts[1]}")
print(f"Non-Fraudulent transactions (Class 0): {class_counts[0]}")
print(f"Percentage of fraudulent transactions: {fraud_percentage:.3f}%")

plt.figure(figsize=(8, 8))
labels = ['Non-Fraud (Class 0)', 'Fraud (Class 1)']
sizes = class_counts.values
colors = ['#66b3ff', '#ff6666']  # Blue for Non-Fraud, Red for Fraud
explode = (0, 0.2)  # 'Explode' the 2nd slice (Fraud) to make it visible

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.3f%%',  # Show percentage with 3 decimal places
        shadow=True, startangle=140)

plt.title('Class Distribution (0: Non-Fraud | 1: Fraud)')
plt.axis('equal')  # Ensures that pie is drawn as a circle.
plt.show()
print("==> INSIGHT: Data is extremely imbalanced.")
print("-" * 30)


# --- 2.5. Analyze PCA Component Correlations ---
print("\n--- 2.5. Analyze PCA Component Correlations ---")

# Correlation with Class Variable
print("Calculating correlations of each PCA component with the 'Class' variable...")
correlations = df.corr()['Class'].copy()
abs_correlations = correlations.abs()

# Filter for only PCA components (V1, V2, etc.)
pca_class_corr = abs_correlations.filter(like='V')
pca_class_corr_sorted = pca_class_corr.sort_values(ascending=False)

# Plot the bar chart
plt.figure(figsize=(15, 8))
sns.barplot(x=pca_class_corr_sorted.index, y=pca_class_corr_sorted.values, palette='vlag')
plt.title('Correlation of PCA Components with Fraud Class (Class=1)', fontsize=16)
plt.xlabel('PCA Component')
plt.ylabel('Correlation')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("-" * 30)


# --- 2.6. Key PCA Component Analysis (Distribution by Class) ---
print("\n--- 2.6. Key PCA Component Analysis (Distribution by Class) ---")

# We already found the sorted components in step 2.4
key_components = pca_class_corr_sorted.head(4).index.tolist()

print(f"Discovered Top 4 Key PCA Components: {key_components}")
print("Plotting their distributions...")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
fig.suptitle('Distribution of Top 4 Key PCA Components by Class', fontsize=20)
axes = axes.flatten()

for i, component in enumerate(key_components):
    sns.kdeplot(df[df['Class'] == 0][component],
                ax=axes[i], label='Non-Fraud (Class 0)', fill=True, alpha=0.4)
    sns.kdeplot(df[df['Class'] == 1][component],
                ax=axes[i], label='Fraud (Class 1)', fill=True, alpha=0.7)

    # Add the original correlation value (not absolute)
    corr_val = correlations[component]
    axes[i].set_title(f'Distribution of {component} (Corr: {corr_val:.2f})')
    axes[i].set_xlabel(component)
    axes[i].set_ylabel('Density')
    axes[i].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
print("==> INSIGHT: The distributions for key components are clearly different for Fraud vs. Non-Fraud, confirming they are predictive.")
print("-" * 30)

print("EDA complete. Proceeding to Preprocessing based on insights.")
print("-" * 30)


# --- 3. Data Preprocessing ---
print("\n--- 3. Data Preprocessing ---")
print("Applying RobustScaler to 'Time' and 'Amount' based on EDA findings...")

# Using RobustScaler as it's less sensitive to outliers
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop the original 'Time' and 'Amount' columns
df.drop(['Time', 'Amount'], axis=1, inplace=True)
print("Original 'Time' and 'Amount' columns dropped.")

# Rearrange columns to put 'Class' at the end
cols = [col for col in df.columns if col != 'Class'] + ['Class']
df = df[cols]
print("'Class' column moved to the end.")

print("\nPreprocessing complete.")
print("-" * 30)

print("\nFINAL: The preprocessed DataFrame 'df' is now ready for modeling.")

df.to_pickle("data/processed_data.pkl") 
