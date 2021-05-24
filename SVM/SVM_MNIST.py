from keras.datasets import mnist
from sklearn import svm
import numpy as np


(train_x, train_y), (test_x, test_y) = mnist.load_data()
#clf=svm.SVC(kernel = 'sigmoid', gamma = 'auto')
#clf=svm.SVC(kernel = 'poly', gamma = 'auto') #avec poly deg=3, C=1 : 97.87%
#clf=svm.SVC(kernel = 'poly', degree = 5, gamma = 'auto', C=10) #96.58
clf=svm.SVC(kernel = 'rbf', gamma = 'auto')
train_x = train_x.reshape((len(train_x),28*28)) #= train_x[i].flatten()
test_x = test_x.reshape((len(test_x),28*28)) #= test_x[i].flatten()
print(train_x,train_y)
clf.fit(train_x,train_y)
svm.SVC()

correct = np.zeros(10)

#for i in range(len(test_x)):
#    if clf.predict(np.array([test_x[i]])) == test_y[i]:
#        correct += 1
#
#print("accuracy : ", correct,"/",len(test_x))

#précision par chiffre


for i in range(len(test_x)):
    if clf.predict(np.array([test_x[i]])) == test_y[i]:
        for j in range(10):
            if test_y[i] == j:
                correct[j] += 1
                print('ok')

for  i in range(10):
   print("accuracy des " +str(i) + " : ", correct[i],"/",len(np.where(test_y==i)))

print("accuracy : ", np.sum(correct),"/",len(test_y))