# Capstone Project - Credit Card Fraud Detection

## Problem Statement

According to the Federal Trade Commission, credit card fraud has been one of the fastest-growing forms of identity theft according. Reports of credit card fraud jumped by 104% from Q1 2019 to Q1 2020, and continue to grow. And the global card losses are expected to exceed $35 billion by 2020 according to the Nilson report. It is evident that there has been a growing amount of financial losses due to credit card frauds as the usage of the credit cards become more and more common.

Frauds are not only costly to the victims, but also to banks and payment networks that issue refunds to consumers. Credit card frauds can be made in many ways such as simple theft, application fraud, counterfeit cards, never received issue (NRI) and online fraud (where the card holder is not present). These fraudulent transactions should be prevented or detected in a timely manner otherwise the resulting losses can be huge. 

Most credit card issuers go above and beyond by offering zero liability for the cardholder, meaning the issuer will refund the entire amount of a fraudulent charge. In order to reduce cost for credit card issuers and loss for cardholders, it is necessary to develope a model that can identify whether a new credit card transaction is fraudulent or not. 

Source: 
*https://www.creditcardinsider.com/blog/2020-fraud-and-identity-theft-analysis/*

## Executive Summary

### Goal

In this project, 2 types of anomaly detection technique will be analyzed in order to find out the best model that has the highest accuracy in identifying whether a credit card transaction is fraudulent or not. 

The class imbalance problem will be addressed and the usage of Synthetic Minority Oversampling Technique (SMOTE) and Random Under sampling (RUS) are applied. 

### 1. Anomaly detection technique: IQR

The interquartile range rule is useful in detecting the presence of outliers. Outliers are calculated by means of the IQR (InterQuartile Range). The first and the third quartile (Q1, Q3) are calculated. An outlier is then a data point that lies outside the interquartile range. 

We will run a single model for each of the following classifiers:

1. Logistic Regression
2. Optimized Logistic Regression
3. Naive Bayes
4. Decision Tree
5. Random Forest 

The interquartile range rule is useful in detecting the presence of outliers.
Outliers are individual values that fall outside of the overall pattern of the rest of the data. The interquartile
range can be used to help detect outliers. 

The steps in calculating interquartile range are
1. Calculate the interquartile range for our data
2. Multiply the interquartile range (IQR) by the number 1.5
3. Add 1.5 x (IQR) to the third quartile. Any number greater than this is a suspected outlier.
4. Subtract 1.5 x (IQR) from the first quartile. Any number less than this is a suspected outlier.

### 2. Anomaly detection technique: Clustering

The objective of any clustering model is to detect patterns in the data. More specifically, it is to group the data into distinct clusters made of data points that are very similar to each other, but distinct from the points in the other clusters. We will explore 2 types of clustering methods:

1. K-means clustering
2. DBSCAN


### Data Source

From Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud. 

The datasets contains transactions made by credit cards in September 2013 by european cardholders. It contains only numerical input variables which are the result of a PCA transformation.

### Performance evaluation
1. Classification report
2. Confusion matrix
3. Precision-Recall Curve
4. ROC Curve and AUC



## Conclusion

In the project, I have explored both supervised and unsupervised learning to detect fraud. I believe we need to be flexible and adjust what techiques to use depending on the datasets.

If we encounter imbalanced datasets in the future like what I have in this project, then we can consider the resampling methods, including SMOTE, Random Under Sampling, and Random Over Sampling. While we are evaluating models, we can also set up different metrics to help us decide which model has the best performance.

The result shows that both Decision Tree and Random Forest models are able to obtain 100% accuracy rate while we use SMOTE, but only 92% and 93% accuracy while we use Random Under Sampling.

Both Decision Tree and Random Forest models are able to obtain 100% accuracy rate. The optimized logistic regression model had a better generalization performance on the testing set with reduced variance as compared to the other models. While the baseline logistic regression had over-fitted, the Naïve Bayes model was unable to achieve higher scores in the classification report and normalized confusion matrix, as compared to the optimized logistic regression model.

The use of optimization for logistic regression had a significant impact on the results with the following two factors being considered:

1. SMOTE to create new synthetic points in order to have a balanced dataset
2. Grid search to select the best hyper-parameters to maximize model performance

Nonetheless, the results were also attributed by the unique strength of logistic regression, where it is intended for binary (two-class) classification problems. It predicts the probability of an instance belonging to the default class, which can be derived as a binary output variable (ie. 0 or 1 classification). By optimizing the logistic regression model, it results in an even better model suited for classification problems.
