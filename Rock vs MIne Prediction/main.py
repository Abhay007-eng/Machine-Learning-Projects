import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('sonar data.csv', header = None)
sonar_data.head()

sonar_data.shape
sonar_data.describe()

sonar_data[60].values_counts()

sonar_data.groupby(60).mean()

X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
print(X, Y)
#print(X) print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)
print(X_train)
print(Y_train)

#logistic regression

model = LogisticRegression()

#training model

model.fit(X_train, Y_train)

#accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on training data:", training_data_accuracy)

#accuracy on test data

X_test_prediction = model.predict(X_train)
test_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on test data:", test_data_accuracy)


#Take input data here
input_data = ()
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
    print("The object is Rock")
else:
    print("The object is Mine")    