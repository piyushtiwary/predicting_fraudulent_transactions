# Predicting Fraudulent Transactions 💳

This repository contains a Jupyter Notebook that demonstrates a machine learning approach to predict fraudulent financial transactions. The project focuses on handling imbalanced datasets, feature engineering, and leveraging an XGBoost classifier for high-performance fraud detection.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features and Engineering](#features-and-engineering)
- [Methodology](#methodology)
- [Results](#results)

## Project Overview

Financial fraud detection is a critical challenge for banks and financial institutions. This project aims to build a robust machine learning model to identify fraudulent transactions, helping to minimize financial losses and protect customers. Due to the highly imbalanced nature of fraud datasets (fraudulent transactions are rare), special techniques like SMOTE (Synthetic Minority Over-sampling Technique) are employed.

The project pipeline involves:
1.  **Data Loading and Sampling:** Handling a large dataset efficiently.
2.  **Exploratory Data Analysis (EDA):** Understanding transaction patterns.
3.  **Feature Engineering:** Creating new features to enhance model performance.
4.  **Handling Class Imbalance:** Using SMOTE to balance the training data.
5.  **Model Training:** Utilizing XGBoost, a powerful gradient boosting algorithm.
6.  **Model Evaluation:** Assessing performance using appropriate metrics (Precision, Recall, F1-Score, ROC AUC, PR AUC).
7.  **Feature Importance Analysis:** Identifying key indicators of fraud.
8.  **Infrastructure Prevention Plan:** Proposing real-world strategies based on model insights.

## Dataset 

The dataset used in this project is `Fraud.csv`, which contains simulated mobile money transactions.
The original dataset is very large (6 million records), so a sampled subset is used for efficient processing while maintaining the integrity of fraud ratios (1 fraudulent transaction for every 100 non-fraudulent ones).[Link](https://drive.google.com/file/d/1TjRG9B4ivxbHQr3f2EqyY0TxMLs2DJYN/view?usp=drive_link)


**Key columns:**
* `step`: Maps a unit of time in the real world. In this dataset, 1 step is 1 hour.
* `type`: Type of online transaction (e.g., CASH_OUT, PAYMENT, TRANSFER).
* `amount`: The amount of the transaction.
* `nameOrig`: Customer ID of the initiator of the transaction.
* `oldbalanceOrg`: Initial balance before the transaction.
* `newbalanceOrig`: New balance after the transaction.
* `nameDest`: Customer ID of the recipient of the transaction.
* `oldbalanceDest`: Initial balance of the recipient before the transaction.
* `newbalanceDest`: New balance of the recipient after the transaction.
* `isFraud`: This is the target variable, indicating a fraudulent transaction (1 = fraud, 0 = not fraud).
* `isFlaggedFraud`: Indicates if the transaction was flagged by the internal system (always TRANSFER and amount > 200,000).

## Features and Engineering

The following features are engineered or processed to improve the model's ability to detect fraud:

### Time-based Features
* `hour`: Hour of the day (0-23). Fraud may occur more often at night.
* `day`: Day of the month (1-31). Look for end-of-month spikes.
* `dayofweek`: Day of the week (0=Monday, 6=Sunday). Weekly fraud trends.
* `is_weekend`: Binary indicator (1 if weekend, 0 otherwise). Fraud may increase during weekends.
* `is_end_of_month`: Binary indicator (1 if `day` >= 30, 0 otherwise). Fraud is heavily skewed toward the last days of the month.

### Transaction Type
* `type_*`: One-hot encoded columns for transaction types (`type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`). The 'type' of transaction is strongly correlated with fraud in this dataset.

### Amount-Based Feature
* `is_high_amount`: Binary indicator (1 if `amount` > 200,000, 0 otherwise). This threshold is significant as it's often a trigger for internal flagging and where fraud frequency starts rising.

### Engineered Features
* `errorOrig`: `oldbalanceOrg - newbalanceOrig - amount`. This detects logical inconsistencies in the sender's balance.
* `errorDest`: `newbalanceDest - oldbalanceDest - amount`. This captures imbalances in the recipient's account.
* `is_full_drain`: `((newbalanceOrig == 0) & (oldbalanceOrg > 0))`. Indicates if the sender's account was fully emptied, a common fraud signature.
* `is_zero_balance`: `(oldbalanceOrg == 0)`. Indicates if the sender's balance was already zero.
* `is_overdrawn`: `(amount > oldbalanceOrg)`. Shows if the transaction amount exceeded the sender's balance.
* `dest_unchanged`: `(oldbalanceDest == newbalanceDest)`. Highlights whether the recipient’s balance remained unchanged, which may point to suspicious/fake recipients.

**Dropped Features:**
* `nameOrig` and `nameDest`: These are nominal, high-cardinality categorical variables with no inherent meaning. They are impossible to generalize to unseen data.
* `isFlaggedFraud`: This feature is redundant as it encodes a simple business rule (TRANSFER and amount > 200,000) that can be inferred from other features. Keeping it might cause data leakage.

## Methodology

1.  **Data Sampling:** The original dataset is downsampled to manage computational resources while preserving the fraud ratio (1 fraud per 100 non-fraud).
2.  **Class Imbalance Handling:** The dataset exhibits severe class imbalance (approx. 99% non-fraud, 1% fraud). To address this, the **SMOTE** (Synthetic Minority Over-sampling Technique) algorithm is applied *only to the training set* after splitting the data. This generates synthetic samples for the minority class (fraud), preventing the model from being biased towards the majority class.
3.  **Model:** An **XGBoost Classifier** (`XGBClassifier`) is used. XGBoost is a powerful and efficient gradient boosting framework known for its excellent performance on tabular data.
4.  **Evaluation Metrics:** Given the imbalance, standard accuracy is misleading. Therefore, the following metrics are prioritized:
    * **Precision:** The proportion of correctly predicted positive observations to the total predicted positives.
    * **Recall (Sensitivity):** The proportion of correctly predicted positive observations to all observations in the actual class.
    * **F1-Score:** The weighted average of Precision and Recall.
    * **ROC AUC Score:** Measures the ability of a classifier to distinguish between classes.
    * **Precision-Recall (PR) AUC:** Particularly useful for imbalanced datasets, it provides a more informative view of performance than ROC AUC when the positive class is rare.

## Results

The XGBoost model, after training with SMOTE, achieved excellent performance in identifying fraudulent transactions.
