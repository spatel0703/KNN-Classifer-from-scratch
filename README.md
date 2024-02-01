# KNN Classifier from scratch for one-hot encoded data

This repository contains a K-Nearest Neighbors (KNN) classifier implemented from scratch for predicting values based on one-hot encoded diabetes data.

## Overview

The project involves the following steps:

### 1. Loading and Preprocessing Data

The diabetes dataset is loaded from a CSV file, and unnecessary columns (Date and Time) are removed. The resulting dataset is saved as 'diabetesdf.csv'. Exploratory Data Analysis (EDA) is performed to understand the dataset's shape, information, tendencies, and unique values in the 'Code' and 'Value' columns.

### 2. One-Hot Encoding

The 'Code' column is converted to one-hot encoding using the `pd.get_dummies` function. The resulting one-hot encoded dataset is saved as 'diabetes_one_hot_dataknn.csv'.

### 3. Data Splitting

The dataset is split into training (60%), validation (10%), and test (30%) sets. The features ('Code' columns) and target variable ('Value') are separated for each set.

### 4. KNN Algorithm Implementation

A custom KNN algorithm is implemented from scratch. The `euclidean_distance` function calculates the Euclidean distance between two instances, and the `knn_predict` function predicts values for the test set based on the nearest neighbors in the training set.

### 5. Evaluation Metrics

The KNN algorithm is evaluated using precision, recall, weighted F1, and macro F1 scores for the first two rows of the test set. The results and metrics are displayed, providing insights into the model's performance.

### 6. K Parameter Optimization

The project includes an exploration of different values of k (number of neighbors) to find the optimal k value. The accuracy for each k value is plotted in the 'k vs Accuracy' graph.

## Usage

To use the KNN classifier, follow the steps outlined in the provided code snippets. Adjust the dataset file paths and hyperparameters as needed. The implementation includes custom functions for data loading, preprocessing, KNN algorithm, and evaluation.

Feel free to explore and experiment with different datasets or modify the code for specific requirements.

**Note: Academic Integrity Warning**

This K-Nearest Neighbors (KNN) classifier implementation is provided for educational and exploratory purposes. While it serves as a valuable resource for understanding the KNN algorithm and its implementation from scratch, it is essential to emphasize that using this code, or portions of it, for school assignments or assessments without proper attribution may constitute plagiarism.

Students are encouraged to use this repository as a learning reference and a starting point for their understanding of KNN algorithms. However, it is strongly advised not to submit this code, modified or unmodified, for academic assignments without explicit permission from their educational institution. Doing so may result in academic penalties, including but not limited to, loss of marks or other disciplinary actions.

Please adhere to your institution's academic integrity policies and guidelines when using this code or any other external resources for academic purposes.

