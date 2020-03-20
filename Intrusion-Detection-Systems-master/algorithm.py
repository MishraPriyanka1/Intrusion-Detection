import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)
from matplotlib import pyplot as plt

# read csv train and test data file from directory using panda library
traindata = pd.read_csv('kddtrain.csv', header=None)
testdata = pd.read_csv('kddtest.csv', header=None)

# selects colum until 41 index
X = traindata.iloc[:,1:42]

# independent train label
Y = traindata.iloc[:,0]

# dependent test label
C = testdata.iloc[:,0]

# test data select until 41 column
T = testdata.iloc[:,1:42]

# fit data single array-like
scaler = Normalizer().fit(X)

# transform matrix for less skewed distribution
trainX = scaler.transform(X)

# operation on a single array-like dataset
scaler = Normalizer().fit(T)

# test data less skewed distribution
testT = scaler.transform(T)

# dependent array- like train data is converted to array using numpy
traindata = np.array(trainX)
trainlabel = np.array(Y)

#array-like test data converted to array using numpy
testdata = np.array(testT)
testlabel = np.array(C)



#traindata = X_train
#testdata = X_test
#trainlabel = y_train
#testlabel = y_test

print("-----------------------------------------LR---------------------------------")
# optimize to get the One Vs Rest scheme using the L-BFGS(analytical derivatives)
model = LogisticRegression(solver='lbfgs', multi_class='ovr')
#train logistic regression model to generate anomaly
model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
#save expected values of test label in a text file
np.savetxt('classical/expected.txt', expected, fmt='%01d')

#output prediction for the input sample
predicted = model.predict(testdata)

# prediction probability of test data
proba = model.predict_proba(testdata)

#save prediction of logistic regression & prediction probability to a text file using numpy
np.savetxt('classical/predictedlabelLR.txt', predicted, fmt='%01d')
np.savetxt('classical/predictedprobaLR.txt', proba)

# assign independent test label to a variable
y_train1 = expected

# independent variable predicted values
y_pred = predicted

#calculate accuracy using train and prediction data of independent y variable
accuracy = accuracy_score(y_train1, y_pred)

# recall is fraction of relevant instances retrieved to total amount of relevant instances calculated using scikit- learn
recall = recall_score(y_train1, y_pred , average="binary")

# precision is fraction of relevant instances among retrieved data using scikit-learn
precision = precision_score(y_train1, y_pred , average="binary")

# average of precision & recall with best value as 1
f1 = f1_score(y_train1, y_pred, average="binary")

#print calculated accuracy, precision, recall and f1 score of logistic regression
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)

#calculated values of accuracy, precision, recall, f1 passed in a variable as array for graph plot
lineararray = [accuracy,precision,recall,f1]


print("-----------------------------------------NB---------------------------------")
# scikit-learn Naive Bayes classification model
model = GaussianNB()

# fit a Naive Bayes model to the data
model.fit(traindata, trainlabel)

print(model)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)

np.savetxt('classical/predictedlabelNB.txt', predicted, fmt='%01d')
np.savetxt('classical/predictedprobaNB.txt', proba)

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)

nbarray = [accuracy,precision,recall,f1]

print("-----------------------------------------DT---------------------------------")

model = DecisionTreeClassifier()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)

np.savetxt('classical/predictedlabelDT.txt', predicted, fmt='%01d')
np.savetxt('classical/predictedprobaDT.txt', proba)
# summarize the fit of the model

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")


print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)

decisionarray = [accuracy,precision,recall,f1]


# summarize the fit of the model
print("--------------------------------------SVM linear--------------------------------------")
y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)

# Plot graph for analysis of various algorithm classification using matplotlib
svmarray = [accuracy,precision,recall,f1]
plt.plot(lineararray,nbarray,decisionarray,svmarray)
plt.show()


