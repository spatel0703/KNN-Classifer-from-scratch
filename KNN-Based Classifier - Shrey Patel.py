#!/usr/bin/env python
# coding: utf-8

# # Loading Data

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

diabetesdf = pd.read_csv('combined_dataset.csv', na_filter = False)

#since we only need the 3rd column(Code), we will remove the Date and Time columns
diabetesdf.drop(diabetesdf.columns[[0,1]], axis = 1, inplace=True)
diabetesdf.to_csv('diabetesdf.csv', index=False)
diabetesdf


# # Exploratory Data Analysis (EDA)

# In[ ]:


print("The shape of the dataset is: ", diabetesdf.shape)
print("The information about the dataset: ", diabetesdf.info())


# In[ ]:


#this gives the datasets tendencies
diabetesdf.describe()


# In[ ]:


#this finds the number of empty points in the dataset
diabetesdf.isnull().sum()


# In[ ]:


#this shows the how many rows of data there are
print("Number of samples in this dataset are: ", len(diabetesdf))

#this finds the unique values among the code column
code_values = diabetesdf['Code'].unique()
print("The unique values among the Code column are:", code_values)

#this finds the unique values among the Value column
unique_values = diabetesdf['Value'].unique()
print("The unique numbers from the Value column are:", unique_values)


# In[ ]:


#this counts the number of occurences for each unique Code value
diabetesdf['Code'].value_counts()


# In[ ]:


diabetesdf['Value'].value_counts()


# In[ ]:


#this graph shows the number of occurences for the Code value on a bar graph

code_counts = diabetesdf['Code'].value_counts()

plt.figure(figsize = (8,6))
plt.bar(code_counts.index, code_counts, color = 'r')
plt.title('Code Number of Occurences')
plt.xlabel('Code')
plt.ylabel('Number of Occurences')
plt.show()


# In[ ]:


import seaborn as sns

#this plot shows the correlation between a Code occurence and its output Value
plt.figure(figsize = (10,8))
sns.stripplot(x = 'Code', y = 'Value', data = diabetesdf.head(300), palette = 'viridis')
plt.title('Code vs Value Stripplot')
plt.xlabel('Code')
plt.ylabel('Value')
plt.show()


# # Converting the 'Code' column to one-hot encoding 

# In[ ]:


#this turns the original dataframe into a one-hot encoded one. Value remains same for each row.
#However, code only returns 1 if true for the Code column
diabetes_one_hot_data = pd.get_dummies(diabetesdf, columns = ['Code'])
diabetes_one_hot_data.to_csv('diabetes_one_hot_dataknn.csv', index=False)
diabetes_one_hot_data


# # 60-10-30% training validation set

# In[ ]:


from sklearn.model_selection import train_test_split

#this splits the training-validation-test sets into a 60-10-30% split
train_data, temp_data = train_test_split(diabetes_one_hot_data, test_size=0.4, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.75, random_state=42)

#this splits the features
X_train = train_data.drop('Value', axis=1)
X_valid = valid_data.drop('Value', axis=1)
X_test = test_data.drop('Value', axis=1)

#this splits the Value variable since it is the target
y_train = train_data['Value']
y_valid = valid_data['Value']
y_test = test_data['Value']

#prints out shape of sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_valid.shape, y_valid.shape)
print("Test set shape:", X_test.shape, y_test.shape)


# In[ ]:


train_data


# In[ ]:


valid_data


# In[ ]:


test_data


# # KNN Algorithm

# In[ ]:


import numpy as np
from tqdm import tqdm

#this calculates the euclidian_distance between two instances
def euclidean_distance(instance1, instance2):
    return np.sqrt(np.sum((instance1 - instance2) ** 2))

#this function takes the training data, Value as the target column, and the test data as inputs with 5 as a hyperparameter
#this will find the nearest neighbors and then predict the value based on those nearest neighbors
def knn_predict(train_data, target_column, data, k=5, subsample_fraction=0.1, num_rows_to_predict=8799):

    test_data_subsample = data.iloc[:num_rows_to_predict, :]
    
    distances = []

    #the reaosn for using tdqm was to track the progress of the predicition since there are a lot of rows
    for row_number, (_, test_instance) in enumerate(tqdm(test_data_subsample.iterrows(), total=len(test_data_subsample), desc="Predicting")):
        current_distances = []

        #this will use the euclidian distance function to calculate the test_data instance and training_data instance
        for i, train_instance in train_data.iterrows():
            distance = euclidean_distance(test_instance[1:], train_instance[1:])
            current_distances.append((distance, train_instance[target_column]))

        #this piece of code will sort the distance determine the nearest neighbor
        current_distances.sort(key=lambda x: x[0])
        neighbors = current_distances[:k]

        counts = {}
        for neighbor in neighbors:
            counts[neighbor[1]] = counts.get(neighbor[1], 0) + 1

        #this will return the row number
        yield counts, int(test_instance[target_column]), row_number

for probs, true_value, row_number in knn_predict(train_data, 'Value', test_data, k=5, subsample_fraction=0.1, num_rows_to_predict=8799):
    predicted_value = max(probs, key=probs.get)
    print(f"Predicted Value: {predicted_value}, True Value: {true_value}, Prediction for row number {row_number}")


# In[ ]:


from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


#I made another knn function to find the accuracy of the data but I only chose the first 2 rows of the test_data to predict
#this is due to the fact that my computer could not handle running so much
def knn_predict(train_data, target_column, data, k_values=[3, 5, 7], subsample_fraction=0.1, num_rows_to_predict=2):
    # Subsample the test data
    test_data_subsample = data.iloc[:num_rows_to_predict, :]
    
    results = {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0}

    k_vs_accuracy = {'k_values': [], 'accuracies': []}

    for k in k_values:
        accuracies = []

        #this code is the same as the last knn predicition function, just uses tqdm for progress tracking
        for row_number, (_, test_instance) in enumerate(tqdm(test_data_subsample.iterrows(), total=len(test_data_subsample), desc=f"Predicting (k={k})")):
            current_distances = []

            for i, train_instance in train_data.iterrows():
                distance = euclidean_distance(test_instance[1:], train_instance[1:])
                current_distances.append((distance, train_instance[target_column]))

            current_distances.sort(key=lambda x: x[0])
            neighbors = current_distances[:k]

            counts = {}
            for neighbor in neighbors:
                counts[neighbor[1]] = counts.get(neighbor[1], 0) + 1

            predicted_value = max(counts, key=counts.get)

            if row_number < 2:
                true_value = int(test_instance[target_column])
                if predicted_value == true_value:
                    results['true_positives'] += 1
                else:
                    results['false_positives'] += 1

            #this will calculate the accuracy of the predictions for the first 2 rows
            accuracies.append(1 if predicted_value == int(test_instance[target_column]) else 0)

        #calculates overall accuracies according to k values
        accuracy = sum(accuracies) / len(accuracies)
        k_vs_accuracy['k_values'].append(k)
        k_vs_accuracy['accuracies'].append(accuracy)

    #this calculates the precision, recall, weighted F1, and macro F1
    true_labels = [str(test_instance[target_column]) for _, test_instance in test_data_subsample.iterrows()]
    predicted_labels = []
    for _, test_instance in test_data_subsample.iterrows():
        current_distances = []

        for i, train_instance in train_data.iterrows():
            distance = euclidean_distance(test_instance[1:], train_instance[1:])
            current_distances.append((distance, str(train_instance[target_column])))

        current_distances.sort(key=lambda x: x[0])
        neighbors = current_distances[:k]

        counts = {}
        for neighbor in neighbors:
            counts[neighbor[1]] = counts.get(neighbor[1], 0) + 1

        predicted_value = max(counts, key=counts.get)
        predicted_labels.append(predicted_value)


    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')
    recall_f1 = f1_score(true_labels, predicted_labels, average='macro')

    print("Results for the first 2 rows:")
    print(results)
    print("\nMetrics for the first 2 rows:")
    print(f"Precision: {precision}, Recall: {recall}, Weighted F1: {weighted_f1}, Recall F1: {recall_f1}")

    return results, k_vs_accuracy

results, k_vs_accuracy = knn_predict(train_data, 'Value', test_data, k_values=[3, 5, 7], subsample_fraction=0.1, num_rows_to_predict=2)

#this plots the k parameter vs accuracy graph
plt.plot(k_vs_accuracy['k_values'], k_vs_accuracy['accuracies'], marker='o')
plt.title('k vs Accuracy')
plt.xlabel('k values')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




