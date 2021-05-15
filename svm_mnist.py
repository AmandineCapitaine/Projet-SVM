from keras.datasets import mnist
from sklearn import svm
import numpy as np


(train_x, train_y), (test_x, test_y) = mnist.load_data()
clf=svm.SVC(kernel = 'poly', gamma = 'auto')
train_x = train_x.reshape((len(train_x),28*28)) #= train_x[i].flatten()
test_x = test_x.reshape((len(test_x),28*28)) #= test_x[i].flatten()
print(train_x,train_y)
clf.fit(train_x,train_y)
svm.SVC()
correct = 0

for i in range(len(test_x)):
    if clf.predict(np.array([test_x[i]])) == test_y[i]:
        correct += 1

print("accuracy : ", correct,"/",len(test_x))