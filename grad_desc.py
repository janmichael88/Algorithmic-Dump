import numpy as np
X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)
X_b = np.c_[np.ones((100,1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
eta = 0.1
m = 100
n_epochs = 50
t0, t1 = 5,50

def learning_schedule(t):
    return(t0 / (t + t1))

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)- yi)
        eta = learning_schedule(epoch*m +i)
        theta = theta - eta*gradients
print(theta)

from sklearn.linear_model import SGDRegressor
import numpy as np 
X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)
sgd_red = SGDRegressor(max_iter=50,penalty=None,eta0=0.1)
sgd_red.fit(X,y.ravel())
print(sgd_red.intercept_,sgd_red.coef_)


#polynomial regression
import numpy as np
m = 100
X = 6*np.random.randn(m,1) - 3
y = X**2 + X + 2 + np.random.randn(m,1)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 

def plot_learning_curves(model,X, y):
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2)
    train_errors,val_errors = [],[]
    #traverse through size of training
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict,y_val))
    plt.plot(np.sqrt(train_errors),"r+",linewidth = 2,label ="train")
    plt.plot(np.sqrt(val_errors),"b-",linewidth=3, label="val")
    plt.show()

lin_reg = LinearRegression()
plot_learning_curves(lin_reg,X,y)