import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
iris = pd.read_csv('iris_data.txt', header=None) #read dataset

# rename each column
iris.columns = ['SepalLengthCm', 'SepalWidthCm', 
              'PetalLengthCm', 'PetalWidthCm', 
              'Species'] 

# shuffle the dataset
iris_visual = shuffle(iris, random_state = 0) 
iris_visual.head(10) #print the top ten entries

# Visualization of the dataset 
import seaborn as sns
sns.set()
sns.pairplot(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']],
             hue="Species", diag_kind="kde")
             
# split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
iris_data = np.array(iris)
X_trn, X_tst, y_trn, y_tst = train_test_split(iris_data[:,0:4], iris_data[:,4],
                                              test_size = 0.4, random_state = 0) 

def gnb_train(X, y, output_label):
    output_size = len(output_label)
    prior = np.zeros(output_size)
    mean = np.zeros((output_size, X.shape[1]))
    std = np.zeros((output_size, X.shape[1]))

    index = 0
    for i in output_label:
        X_index = X[np.where(y == i)]
        X_mean = np.mean(X_index, axis = 0, keepdims = True)
        X_var = np.var(X_index,axis = 0, keepdims = True)
        
        mean[index] = mean[index] + X_mean
        std[index] = std[index] + X_var
        prior[index] = prior[index] + (X_index.shape[0] / X_trn.shape[0])
        index = index + 1


    return prior, mean, std

def gnb_predict(X, prior, mean, std, output_label):
    predict = []

    eps = 1e-10 # In order to avoid 0 denominator
    
    for i in X:
        x_sample = np.tile(i,(len(output_label),1)) 
        numerator = np.exp(-(x_sample.astype(float) - mean) ** 2 / (2 * std + eps))
        denominator = np.sqrt(2 * np.pi * std + eps)
        log_list = np.sum(np.log(numerator/denominator),axis=1)
        prior_x_class = log_list + np.log(prior)             
        prior_x_class_index = np.argmax(prior_x_class)                 
        x_class = output_label[prior_x_class_index]                 
        predict.append(x_class)        
     
    return predict
    
# Testing the model
from sklearn.metrics import hamming_loss
output_label = list(set(y_trn))
a = []

prior, mean, std = gnb_train(X_trn, y_trn, output_label)
y_pred = gnb_predict(X_tst, prior, mean, std, output_label)
error = hamming_loss(y_tst, y_pred)
print("Testin Accuracy", 1-error)


