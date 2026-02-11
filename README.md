
## Telco Customer Churn Prediction  

---

# a. Problem Statement

Customer churn prediction is a critical business problem in the telecom industry. Retaining existing customers is significantly more cost-effective than acquiring new ones.

The objective of this project is to build and compare multiple machine learning classification models to predict whether a customer will churn (leave the service) based on demographic, account, and service usage features.

---

# b. Dataset Description

## Dataset Information

- **Dataset Name:** Telco Customer Churn Dataset  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
- **Problem Type:** Binary Classification  

---

## Dataset Characteristics

- **Total Instances:** 7,043  
- **Total Features:** 20+ features  
- **Target Column:** `Churn`  

---

## Data Preprocessing Steps

- Removed `customerID`
- Converted `TotalCharges` to numeric format
- Handled missing values
- Label encoding of binary columns
- One-hot encoding of categorical features
- Feature scaling using `StandardScaler`
- Train-Test split (80% / 20%) with stratification

---

# c. Models Implemented

The following six machine learning classification models were implemented on the same dataset:

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbours (kNN)  
- Gaussian Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

---

# Model Performance Comparison

| ML Model            | Accuracy | Precision | Recall | F1 Score | ROC-AUC | MCC   |
|--------------------|----------|-----------|--------|----------|---------|-------|
| Logistic Regression | 0.740 | 0.507 | 0.789 | 0.617 | 0.842 | 0.459 |
| Decision Tree       | 0.739 | 0.505 | 0.765 | 0.609 | 0.827 | 0.445 |
| KNN                 | 0.751 | 0.531 | 0.524 | 0.528 | 0.752 | 0.358 |
| Naive Bayes         | 0.659 | 0.430 | **0.866** | 0.574 | 0.811 | 0.399 |
| Random Forest       | **0.778** | 0.567 | 0.693 | 0.623 | **0.843** | 0.473 |
| XGBoost             | 0.761 | 0.535 | 0.757 | **0.627** | 0.835 | **0.473** |

---

# Model Performance Observations

## Logistic Regression  
Logistic Regression achieved strong recall (0.789) among linear models and a high ROC-AUC (0.842), indicating good class separability. Although precision (0.507) is moderate, the model effectively identifies churn customers. It serves as a strong baseline model.

## Decision Tree  
The Decision Tree model achieved accuracy (0.739) similar to Logistic Regression but slightly lower MCC (0.445), indicating higher variance. While it captures nonlinear patterns, it is more prone to overfitting.

## K-Nearest Neighbours (kNN)  
kNN achieved moderate accuracy (0.751) but lower recall (0.524) and the lowest MCC (0.358). It is sensitive to feature scaling and struggles with class imbalance in churn datasets.

## Naive Bayes  
Naive Bayes produced the **highest Recall (0.866)** among all models, meaning it detects most churn cases. However, precision (0.430) is low, leading to more false positives.

## Random Forest (Ensemble)  
Random Forest achieved the **highest Accuracy (0.778)** and **highest ROC-AUC (0.843)**, demonstrating strong generalization. It provides a balanced trade-off between precision and recall and reduces overfitting compared to Decision Tree.

## XGBoost (Ensemble)  
XGBoost achieved the **highest F1 Score (0.627)** and **highest MCC (0.473)**, indicating the best overall balance between precision and recall. It captures complex feature interactions effectively and performs best overall for churn prediction.

---

# Conclusion

This project demonstrates that ensemble methods outperform standalone classifiers for Telco churn prediction.

While Random Forest achieved the highest accuracy and AUC, XGBoost provided the best balance between precision and recall as reflected by F1 Score and MCC.

Therefore, **XGBoost is the most suitable model for predicting customer churn in this dataset.**
