# Fake_news_detection
Overview
This project aims to detect fake news using machine learning techniques. It utilizes a dataset containing both fake and true news articles, applying natural language processing (NLP) and machine learning algorithms to classify news articles as fake or real.


Technologies Used:

Python

Pandas

NumPy

Scikit-learn

NLTK

TfidfVectorizer

Joblib (for model saving)

Dataset
The dataset consists of two separate files: one containing true news articles and another with fake news articles, sourced from Kaggle. The true news dataset contains 21,417 articles, while the fake news dataset contains 23,481 articles. Each dataset includes a label column, where 1 indicates fake news and 0 indicates true news. Both datasets are combined for analysis.

Data Cleaning
Text data must be cleaned to remove unusable words, special symbols, and other elements that may hinder the machine learning model's ability to detect patterns. The following preprocessing steps are applied:

Lemmatization: Converts words to their base forms (e.g., "stays" to "stay").
Stop Word Removal: Filters out common words that provide little value (e.g., "the," "and").
Regular Expression Cleaning: Removes special characters and numbers.
Modeling
Train-Test Split: The dataset is split into 80% training data and 20% testing data using the train_test_split function from Scikit-learn.
Tfidf Vectorization: The text data is transformed into numerical features using the TfidfVectorizer.
Multinomial Naive Bayes Classifier: A Multinomial Naive Bayes Classifier is employed for classification, which is particularly effective for text classification tasks.
Metrics
Model performance is evaluated using several metrics:

The model achieved an accuracy of 93% on the training dataset; however, the prediction accuracy on the test dataset is lower at approximately 57%. This discrepancy in performance is primarily due to the limited amount of data used during the training phase. For this project, only a subset of the available data was utilized, which may have affected the model's ability to generalize effectively to unseen data.

