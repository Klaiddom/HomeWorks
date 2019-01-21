from pp import *
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn import model_selection as ms
from sklearn import svm

#Dataset was initally pre-processed in sublime text for replacing '?' to nan

data = pd.read_csv('DatasetWithMissingValues.txt', header=None)

#Initial Exploration
print (data.head())
print (data.describe())
print (data.info())
print (data.dtypes)

#Scaled data
clean1 = Scale(ReplaceWithMode(data))
train, test = ms.train_test_split(clean1, test_size=0.2)
x_train = train.loc[:, [0,1,2,3,4]]
y_train = np.array(train.loc[:,[5]])
x_test = test.loc[:, [0,1,2,3,4]]
y_test = np.array(test.loc[:,[5]])
clfs = svm.SVC()
clfs.fit(x_train, y_train)
print ('Scaled data')
print (clfs.score(x_test, y_test))

#Without data scale
clean2 = ReplaceWithMode(data)
train, test = ms.train_test_split(clean2, test_size=0.2)
x_train = train.loc[:, [0,1,2,3,4]]
y_train = np.array(train.loc[:,[5]])
x_test = test.loc[:, [0,1,2,3,4]]
y_test = np.array(test.loc[:,[5]])
clfns = svm.SVC()
clfns.fit(x_train, y_train)
print ('Without data scaling')
print (clfns.score(x_test, y_test))

#Standartization data
clean = Standartization(ReplaceWithMode(data))
train, test = ms.train_test_split(clean, test_size=0.2)
x_train = train.loc[:, [0,1,2,3,4]]
y_train = np.array(train.loc[:,[5]])
x_test = test.loc[:, [0,1,2,3,4]]
y_test = np.array(test.loc[:,[5]])
clfst = svm.SVC()
clfst.fit(x_train, y_train)
print ('Standartization')
print (clfst.score(x_test, y_test))

#Median
clean = ReplaceWithMedian(data)
train, test = ms.train_test_split(clean, test_size=0.2)
x_train = train.loc[:, [0,1,2,3,4]]
y_train = np.array(train.loc[:,[5]])
x_test = test.loc[:, [0,1,2,3,4]]
y_test = np.array(test.loc[:,[5]])
clfmed = svm.SVC()
clfmed.fit(x_train, y_train)
print ('Median')
print (clfmed.score(x_test, y_test))

#Average
clean = ReplaceWithAverage(data)
train, test = ms.train_test_split(clean, test_size=0.2)
x_train = train.loc[:, [0,1,2,3,4]]
y_train = np.array(train.loc[:,[5]])
x_test = test.loc[:, [0,1,2,3,4]]
y_test = np.array(test.loc[:,[5]])
clfa = svm.SVC()
clfa.fit(x_train, y_train)
print ('Average')
print (clfa.score(x_test, y_test))