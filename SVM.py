import pandas as pd 
import os
import matplotlib.pyplot as plt 
#set working directory
os.getcwd()
path = '/Users/janmichaelaustria/Documents/Python/Random Projects'
os.chdir(path)
os.getcwd()

iris = pd.read_csv('iris.csv')
#iris = iris.drop(['variety'],axis=1)

target = iris['variety']

#create empty set
s = set()

for val in target:
	s.add(val)

rows = list(range(100,150))
iris = iris.drop(iris.index[rows])

#grab sepal/petal length

x = iris['sepal.length']
y = iris['petal.length']

setosa_x = x[:50]
setosa_y = y[:50]

versicolor_x = x[50:]
versicolor_y = x[50:]

plt.figure(figsize = (8,6))
plt.scatter(setosa_x,setosa_y,marker = "+",color="green")
plt.scatter(versicolor_x,versicolor_y,marker='_',color='red')
plt.show()

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np 

#drop the rest of the features and extra the target values
iris = iris.drop(['sepal.width','petal.width'],axis=1)
Y = []
target = iris['variety']

for bah in target:
	if bah == "Setosa":
		Y.append(-1)
	else:
		Y.append(1)

iris = iris.drop(['variety'],axis=1)
X = iris.values.tolist()

##shuffle and split the data into training and test set
X,Y = shuffle(X,Y)
x_train = []
y_train = []
x_test = []
y_test = []

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size = 0.8)

#conver to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

#reshape arrays
y_train = y_train.reshape(80,1)
y_test = y_test.reshape(20,1)

#build svm model 
import numpy as np
train_1 = x_train[:,0]
train_2 = x_train[:,1]
#reshape
train_1 = train_1.reshape(80,1)
train_2 = train_2.reshape(80,1)

#initialize weights and learning rate
w1 = np.zeros((80,1))
w2 = np.zeros((80,1))
epochs = 1
alpha = 0.0001

while(epochs<10000):
	y = w1*train_1 + w2*train_2
	prod = y*y_train
	print(epochs)
	#set count 
	count = 0
	for val in prod:
		if (val >=1):
			cost = 0
			#weigfht update for 1 and 2 under the first condition
			w1 = w1 - alpha*(2*1/epochs * w1)
			w2 = w2 - alpha*(2*1/epochs * w1)
		else:
			#weight update for second
			cost = 1 - val
			w1 = w1 + alpha*(train_1[count]*y_train[count] - 2*1/epochs*w1)
			w2 = w2 + alpha*(train_2[count]*y_train[count] - 2*1/epochs*w2)
		#adjust count
		count += 1
	epochs += 1

from sklearn.metrics import accuracy_score

#clip the weights
index = list(range(20,80))
w1 = np.delete(w1,index)
w2 = np.delete(w2,index)

#reshape again
w1 = w1.reshape(20,1)
w2 = w2.reshape(20,1)

#extract the test data features
test_1 = x_test[:,0].reshape(20,1)
test_2 = x_test[:,1].reshape(20,1)

#predict
y_pred = w1*test_1 + w2*test_2
predictions = []
for val in y_pred:
	if (val>1):
		predictions.append(1)
	else:
		predictions.append(-1)
print(accuracy_score(y_test,predictions))



















































