# 💬 Customer Query Routing System

An AI-powered customer query routing system that automatically classifies customer-service queries into the most relevant category using Natural Language Processing (NLP) and Machine Learning.

## 📌 Project Overview

Customer-service teams receive a large number of queries every day. Manually identifying the correct category for each query can take time and may lead to inconsistent routing.

This project uses Machine Learning to automatically identify the intent of a customer query and route it to the appropriate customer-service category.

The project uses:

- TF-IDF for text feature extraction
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Streamlit for the user interface

## 🎯 Objective

The main objective is to build a multi-class classification system that can:

1. Accept a customer query as text.
2. Convert the text into numerical features using TF-IDF.
3. Predict the most relevant customer-service category.
4. Display the predicted category through a simple Streamlit application.

## 📊 Dataset

The project uses the **BANKING77** dataset, which contains customer-service queries belonging to multiple banking-related intent categories.

The dataset is divided into:

- Training data: 10,003 records
- Test data: 3,080 records
- Number of categories: 77

The dataset contains two main columns:

- `text` — Customer query
- `category` — Customer query category

Dataset source: BANKING77 dataset on Hugging Face.

## 🔍 Data Preparation

The following steps were performed:

- Loaded the training and testing datasets.
- Checked the dataset structure.
- Checked for missing values.
- Checked for duplicate records.
- Examined the distribution of categories.
- Converted text into numerical features using TF-IDF.

No missing values or duplicate records were found in the training dataset.

## 🧠 Machine Learning Models

Three classification models were evaluated:

### 1. Logistic Regression

Accuracy: **85.91%**

Weighted F1-score: **85.82%**

### 2. Multinomial Naive Bayes

Accuracy: **79.87%**

Weighted F1-score: **78.52%**

### 3. Linear SVM

Accuracy: **88.99%**

Weighted F1-score: **88.99%**

The Linear SVM model was selected as the final model because it achieved the highest performance among the tested models.

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The Linear SVM achieved approximately **88.99% accuracy** on the test dataset.

## 🧪 Example Predictions

Example customer queries tested with the trained model:

| Customer Query | Predicted Category |
|---|---|
| I have been waiting for my card for two weeks | `card_arrival` |
| Someone used my card and I don't recognize the payment | `compromised_card` |
| I forgot my PIN | `pin_blocked` |
| Why was I charged a fee for withdrawing cash? | `cash_withdrawal_charge` |
| My contactless payment is not working | `contactless_not_working` |

## 🖥️ Streamlit Application

The project includes a Streamlit web application where users can enter a customer query and receive the predicted category.

### Run the application

From the `app` directory:

```bash
python -m streamlit run app.py