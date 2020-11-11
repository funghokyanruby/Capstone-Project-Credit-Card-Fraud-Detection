# Capstone Project - Credit Card Fraud Detection

## Problem Statement

According to the Federal Trade Commission, credit card fraud has been one of the fastest-growing forms of identity theft according. Reports of credit card fraud jumped by 104% from Q1 2019 to Q1 2020, and continue to grow. And the global card losses are expected to exceed $35 billion by 2020 according to the Nilson report. It is evident that there has been a growing amount of financial losses due to credit card frauds as the usage of the credit cards become more and more common.

Frauds are not only costly to the victims, but also to banks and payment networks that issue refunds to consumers. Credit card frauds can be made in many ways such as simple theft, application fraud, counterfeit cards, never received issue (NRI) and online fraud (where the card holder is not present). These fraudulent transactions should be prevented or detected in a timely manner otherwise the resulting losses can be huge. 

Most credit card issuers go above and beyond by offering zero liability for the cardholder, meaning the issuer will refund the entire amount of a fraudulent charge. In order to reduce cost for credit card issuers and loss for cardholders, it is necessary to develope a model that can identify whether a new credit card transaction is fraudulent or not. 

Source: 
*https://www.creditcardinsider.com/blog/2020-fraud-and-identity-theft-analysis/*

## Executive Summary

In this project, 2 types of anomaly detection techniques will be analyzed in order to find out the best model that has the highest accuracy in identifying whether a credit card transaction is fraudulent or not and these 2 techniques demonstrate both supervised and unsupervised learning techniques to detect fraud. 

Since the dataset is imbalanced, it will be addressed by making use of Synthetic Minority Oversampling Technique (SMOTE) and Random Under sampling (RUS). 

### 1. Supervised learning technique: IQR + Classification

The interquartile range rule is useful in detecting the presence of outliers. Outliers are calculated by means of the IQR (InterQuartile Range). The first and the third quartile (Q1, Q3) are calculated. An outlier is then a data point that lies outside the interquartile range. 

After the removal of outliers, we will run a single model for each of the following classifiers:

1. Logistic Regression
2. Optimized Logistic Regression
3. Naive Bayes
4. Decision Tree
5. Random Forest 

The 4 matrics below will be used to evaluate the performance of the 5 classifiers:
1. Classification report
2. Confusion matrix
3. Precision-Recall Curve
4. ROC Curve and AUC

Using known frad cases to train a model to recognise new cases, we will be able to find out which classificaiton model has the best performance in accurately detecting fraud. 

### 2. Unsupervised learning technique: Clustering

In reality, not all datasets have reliable data labels. Hence, it is also necessary to be able to use unsupervised learning techniques to detect fraud. When using unsupervised learning techniques for fraud detection, you want to distinguish normal from abnormal (thus potentially fraudulent) behavior. However, due to the PCA transformed dataset, it is not feasible to understand the characteristics and features of the data.

The objective of any clustering model is to detect patterns in the data. More specifically, it is to group the data into distinct clusters made of data points that are very similar to each other, but distinct from the points in the other clusters. We will explore 2 types of clustering methods:

1. K-means clustering
2. DBSCAN


### Data Source

From Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud. 

The datasets contains transactions made by credit cards in September 2013 by European cardholders. It contains only numerical input variables which are the result of a PCA transformation.

### Goal

The project has covered 2 anomaly detection techniques, 2 types of resampling methods, both supervised and unsupervised learning appraoches and I am sure this result will help our primary stakeholders - credit card issuers, and our secondary stakeholers - credit card holders to reduce loss by developing models that can identify whether a new credit card transaction is fraudulent or not.


## Conclusion

In the project, I have explored both supervised and unsupervised learning to detect fraud. I believe we need to be flexible and adjust what techiques to use depending on the datasets.

### 1. Supervised learning technique: IQR + Classification

After applying SMOTE to address the imbalance dataset problem and removal of outliers using IQR technique, we have run the following 5 classifiers:

1. Logistic Regression
2. Optimized Logistic Regression
3. Naive Bayes
4. Decision Tree
5. Random Forest 

Both Decision Tree and Random Forest models are able to obtain 100% accuracy rate for train dataset. The optimized Logistic Regression model shows a huge improvement in generalization performance on the testing set with reduced variance as compared to the baseline Logistic Regression model. While the baseline Logistic Regression had over-fitted, the Naïve Bayes model is unable to achieve higher scores in the classification report and normalized confusion matrix, as compared to the optimized logistic regression model.

The use of optimization for logistic regression had a significant impact on the results with the following two factors being considered:

1. SMOTE to create new synthetic points in order to have a balanced dataset
2. Grid search to select the best hyper-parameters to maximize model performance

Nonetheless, the results were also attributed by the unique strength of Logistic Regression, where it is intended for binary classification problems. It predicts the probability of an instance belonging to the default class, which can be derived as a binary output variable (ie. 0 or 1 classification). By optimizing the logistic regression model, it results in an even better model suited for classification problems.

However, the last two models Decision Tree and Random Forest bring us a even more promising result that they are both able to obtain 100% accuracy rate for train set, without Grid search. I believe it is possible to hit an almost perfect 100% accuracy rate if using Grid search to select the hyper-parameters for Decision Tree and Random Forest. When looking at Precision-Recall Curve and ROC Curve and AUC, the ideal point is therefore the top-left corner of the ROC plot where false positives are zero adn we are still able to come to conclusion that Random Forest performs the best among all. 

Last but not least, SMOTE only works well if the minority case features are similar. For the particular dataset that we are using right now, the fraud case features are PCA-transformed. Hence, we are not sure if the fraud cases are spread through the data and similar. Hence, we will also apply another resampling method, Random Under Sampling to make some comparisons.  

### Comparison between SMOTE and RUS Accuracy rate

|**Classifier**|SMOTE - Accuracy rate(%)|RUS - Accuracy rate(%)|
| :---|:---|:---|
|**Baseline Logistic Regression**|Train 99.92, Test 99.91|Train 99.92, Test 99.91|
|**Optimized Logistic Regression**|Train 98.75, Test 98.78|Train 98.17, Test 97.65|
|**Naive Bayes**|Train 90.28, Test 90.37|Train 91.38, Test 91.11|
|**Decision Tree**|Train 100, Test 99.85|Train 100, Test 93.88|
|**Random Forest**|Train 100, Test 99.98|Train 100, Test 96.11|

Compare to the result with SMOTE techique, the performance of the 5 models are indeed very similar when using Random Unders Sampling method. 

There are some differences in the performance of Decision Tree and Random Forest. When looking at Precision-Recall Curve and ROC Curve and AUC, the top-left corner of plot of Decision Tree and Random Forest models are no longer zero. Also, the accuracy rate of these 2 models both drop from 99% to 93% and 96% on test data only. 

Nonetheless, both SMOTE and Random Under Sampling methods are able to lead to the same conclusion that Random Forest model has the best performance among 5 classifiers. 


### 2. Unsupervised learning technique: Clustering

In reality, not all datasets have reliable data labels. Hence, it is also necessary to be able to use unsupervised learning techniques to detect fraud. The objective of any clustering model is to detect patterns in the data. More specifically, it is to group the data into distinct clusters made of data points that are very similar to each other, but distinct from the points in the other clusters. We have explored 2 types of clustering methods:

1. K-means clustering
2. DBSCAN


#### K-means clustering 

The objective of k-means is to minimize the sum of all distances between the data samples and their associated cluster centroids. I have implemented MiniBatch K-means with 8 clusters, and applied Elbow method and see what the optimal number of clusters should be based on this method.

The result shows that: 

Recall or Sensitivity: True Positives/(True Positives + False Negatives) i.e what percentage of fraud is correctly identified is 85%. 

Also, ROC score is 0.9, which denotes a pretty excellent result.

#### DBSCAN

his technique is based on the DBSCAN clustering method. DBSCAN is a non-parametric, density based outlier detection method in a one or multi dimensional feature space.

In the DBSCAN clustering technique, all data points are defined either as Core Points, Border Points or Noise Points.
- Core Points are data points that have at least MinPts neighboring data points within a distance ℇ.
- Border Points are neighbors of a Core Point within the distance ℇ but with less than MinPts neighbors within the distance ℇ.
- All other data points are Noise Points, also identified as outliers.

Outlier detection thus depends on the required number of neighbors MinPts, the distance ℇ and the selected distance measure, like Euclidean or Manhattan.

Compare to K-means Clustering, the number of clusters does not need to be predefined when using DBSCAN. The algorithm finds core samples of high density and expands clusters from them. DBSCAN works well on data containing clusters of similar density. It can be used to identify fraud as very small clusters.

Since the size of the particular dataset we are using now is too large for DBSCAN to run, I split dataset into train and test set, and use test set as the subsample of the dataset. 

The result shows that DBSCAN has 100% accuracy in detecting fraud.


## Recommendations

### 1. Grid search CV

When using classifiers to flag fraud transactions, the use of Grid search CV to optimize Logistic regression has shown a huge improvement in generalization performance on the testing set with reduced variance as compared to the baseline Logistic Regression model.

The optimized logistic regression model had performed well with a decrease in the Type 2 Error: False Negatives (predicted non-fraudulent but actually a fraudulent transaction). A remarkable decrease of 76% was obtained from a score of 0.42 to 0.1, when comparing the results for the optimized logistic regression model against its baseline model.

However, the Type 1 Error: False Positives (predicted fraudulent but actually a non-fraudulent transaction) had increased by 3%, from 0 to 0.03, by comparing the optimized logistic regression with its baseline model.

Overall, the impact of having more false positives was mitigated with a notable decrease in false negatives. This is important in a credit card fraud prediction scenario where the risk of misidentifying a non-fraudulent transaction outweighs the potential of missing out genuine fraudulent transactions.

With a good outcome of the test values predicted classes as compared to their actual classes, the normalized confusion matrix results for the optimized logistic regression model had outperformed the other models.

We have only carried out hyperparameter tuning for logistic regression model in this project. In the future, I see that hyperparameter tuning with Grid Search will also be able to improve th performance of other classification models. 


### 2. PCV-transformed dataset

When using unsupervised learning techniques for fraud detection, we want to distinguish normal from abnormal (thus potentially fraudulent) behavior. However, due to the PCA transformed dataset, it is not feasible to understand the characteristics and features of the data.

The result derived from this project may not be totally applicable to other datasets, due to the limitation of the PCA-transformed features.

However, I suggest that we can use another angle to see this problem. Though we are unable to study the correlation between features and class distribution for this particular datset, the project has covered 2 anomaly detection techniques, 2 types of resampling methods, both supervised and unsupervised learning appraoches and I am sure this result will help our primary stakeholders - credit card issuers, and our secondary stakeholers - credit card holders to reduce loss by developing models that can identify whether a new credit card transaction is fraudulent or not.
