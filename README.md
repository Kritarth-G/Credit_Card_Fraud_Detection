# Credit Card Fraud Detection Project

This project uses machine learning to detect fraudulent credit card transactions from a public Kaggle dataset. The model pipeline includes EDA, feature engineering, and a comparison of three models: Logistic Regression (Baseline), Tuned Random Forest (Intermediate), and Tuned XGBoost (Advanced).

Project structure
.
├── data/
   │ └── creditcard.csv (150MB dataset, managed by Git LFS)
   │ ├── dataPreprocess.py (Data preparation, scaling, train/test split)
├── Models/
  │ ├── logisticRegression.py
  │ ├── randomForest.py
  │ └── XGBoost.py (Contains the two-stage tuning logic)
├── README.md (This file)
├── requirements.txt

🚀 Steps to Use the Repository
This guide assumes the user is familiar with the command line and has Git installed.

1.  Initial Setup (Install Git LFS)
Because the creditcard.csv file is 150 MB, the user must install Git Large File Storage (LFS) first to correctly download the data.
Install LFS: (Mac/Linux users often use brew install git-lfs)
Initialize LFS:

Bash
git lfs install

2.  Clone & Prepare
Clone the repository to a local machine and install the necessary Python environment.
Clone the Repository:
Bash
git clone [YOUR_REPOSITORY_URL]
cd [REPO_NAME]

Install Dependencies:

Bash

pip install -r requirements.txt

3. 💾 Data Preparation (Assumed Script)
The user must run the data preparation step to create the necessary variables (X_train, y_train, etc.) required by the model scripts.

Run Data Preprocessing: *(Since this script isn't in your Models/ folder, the user must run your notebook or separate data prep script here). Example placeholder (adjust based on your actual file):

Bash

python data_prep_script.py 
4. 📈 Execute Model Tuning
Run the scripts in the Models/ directory sequentially to train, tune, and evaluate each model.

Run Baseline (Logistic Regression):

Bash

python Models/logisticRegression.py
Run Intermediate (Random Forest):

Bash

python Models/randomForest.py
Run Advanced (XGBoost Tuning):

Bash

python Models/XGBoost.py
