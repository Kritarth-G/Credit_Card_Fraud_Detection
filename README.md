# Credit Card Fraud Detection Project

This project uses machine learning to detect fraudulent credit card transactions from a public Kaggle dataset. The model pipeline includes EDA, feature engineering, and a comparison of three models: Logistic Regression (Baseline), Tuned Random Forest (Intermediate), and Tuned XGBoost (Advanced).

Project Structure
Payment-Fraud-Detection-using-Graph-Neural-Networks/
├── data/                           # Data directory
│   └── creditcard.csv (150MB dataset, managed by Git LFS)
│   ├── dataPreprocess.py (Data preparation, scaling, train/test split)
├── models                   
│   ├── logisticRegression.py
│   ├── randomForest.py
│   └── XGBoost.py (Contains the two-stage tuning logic)
├── requirements.txt
└── README.md                     # This file

Installation
-> Git LFS is the essential tool for managing your large 150 MB creditcard.csv dataset, as standard Git cannot handle files exceeding 100 MB. Git LFS tracks the large file, stores its data separately from the repository, and commits only a small pointer to GitHub, keeping your project history fast and functional.
-> Clone the repository
