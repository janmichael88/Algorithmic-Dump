#grab the imports 
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

#assign label variabels and featrues
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#in
le = preprocessing.LabelEncoder()

#encode weather
weather_encoded = le.fit_transform(weather)

#encdoe all three
features = [weather,temp,play]
list_encoded_features = []
for i in features:
	encoded_feature = le.fit_transform(i)
	list_encoded_features.append(encoded_feature)

#zip features
features = zip(list_encoded_features[0],list_encoded_features[1])
features = np.transpose(np.vstack((list_encoded_features[0],list_encoded_features[1])))

#generate the model 
#instantiage gaussain classifier
model = GaussianNB()
#fit the model
model.fit(features,list_encoded_features[2])
#predict a test point
predicted = model.predict([[0,2]])
print(predicted)

#load dataset
#grab the imports 
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
wine = datasets.load_wine()
print(wine.feature_names)
print(wine.target_names)
print(wine.data.shape)

#split data into train test split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) 
# 70% training and 30% test
#Create a Gaussian Classifier
gnb = GaussianNB()
#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))












