import numpy as numpy
import matplotlib.pyplot as plt
cur_x = 3
#learning rate
rate = 0.01
#when to stop algorithm
precision = 0.000001
previous_step_size = 1
max_iters = 10000
#start the iter counter
iters = 0
#gradient of the function
df = lambda x: 2*(x+5) 

while previous_step_size > precision and iters < max_iters:
	#store value
	prev_x = cur_x
	#adujust
	cur_x = prev_x - rate*df(prev_x)
	#actual change in x
	previous_step_size = abs(cur_x-prev_x)
	#adjust the count
	iters = iters+1
	print("Iteration",iters,"\nX value is",cur_x)

print("The local minimum occurs at",cur_x)


#this is the function we want to minimuze for a selction of thetas
def cal_cost(theta,X,y):
	'''
	calculate cost for given X and y
	X = Row of X's np.zeros((2,j))
	y = actual y's np.zeros((2,l))
	j is the number of features
	'''
	m = len(y)
	predictions = X.dot(theta)
	cost = (1/2*m)*np.sum(np.square(predictions-y))
	return(cost)

def gradient_descent(X,y,theta,learning_rate = 0.01,iterations = 200):
	'''
	X = matrix of X with added bias units
	y = vector of y
	theta = Vector of thetas np.random(j,l)
	returns final theta vector and array of cost history for each iterations
	'''
	m = len(y)
	cost_history = np.zeros(iterations)
	theta_history = np.zeros((iterations,2))
	for itr in range(iterations):
		#get prediction
		prediction = np.dot(X,theta)
		#get error
		error = prediction - y
		#adjust theta
		theta = theta - (1/m)*learning_rate*(X.T.dot(error))
		#store thetas
		theta_history[itr,:] = theta.T
		#store cost
		cost_history[itr] = cal_cost(theta,X,y)

	#get back vars
	return(theta,cost_history,theta_history)

X = 2*np.random.rand(100,1)
y = 4 + 3*X+np.random.randn(100,1)
plt.scatter(X,y)
plt.show()
lr  = 0.01
n_iter = 1000
theta = np.random.randn(2,1)

X_b = np.c_[np.ones((len(X),1))*2,X]
theta,cost_history,theta_history = gradient_descent(X_b,y,theta,lr,n_iter)
print('Theta0:{:0.3f},\nTheta1:{:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))

#plot the cost histroy over iterations
fig,ax = plt.subplots(figsize=(12,8))
ax.set_ylabel("J(theta)")
ax.set_xlabel("iterations")
_=ax.plot(range(n_iter),cost_history,'b.')
plt.show()

#zoom in to 200
fig,ax = plt.subplots(figsize=(10,8))
_=ax.plot(range(200),cost_history[:200],"r.")
plt.show()

#create function to show the affecting of choosing iters and learning rates
def plot_GD(n_iter,lr,ax,ax1=None):
	#init the plot
	#X and y are already created 
	_=ax.plot(X,y,"b.")
	theta = np.random.randn(2,1)

	#set transparency for plotting
	tr = 0.1
	cost_history = np.zeros(n_iter)
	for i in range(n_iter):
		#get initial predictions
		pred_prev = X_b.dot(theta)
		#only for 1 iteration, loop through this function in the next pass
		theta,h,_ = gradient_descent(X_b,y,theta,lr,1)
		pred = X_b.dot(theta)
		#update cost history
		cost_history[i] = h[0]

		if (i%25 == 0):
			_=ax.plot(X,pred,"r-",alpha = tr)
			if tr < 0.8:
				tr = tr + 0.2
		if not ax1 == None:
			_=ax1.plot(range(n_iter),cost_history,"b.")

#plot the graphs for different iterations and learning rate combinations
fig = plt.figure(figsize=(40,30),dpi=200)
fig.subplots_adjust(hspace = 0.4,wspace=0.4)

#list of iterations and learning rates
it_lr = [(2000,0.001),(500,0.01),(200,0.05),(100,0.1)]
count = 0
for n_iter, lr in it_lr:
	count += 1

	#plot first figure
	ax = fig.add_subplot(4,2,count)
	count += 1
	#plot second figure
	ax1 = fig.add_subplot(4,2,count)
	ax.set_title("lr:{}".format(lr))
	ax1.set_title("iterations:{}".format(n_iter))
	plot_GD(n_iter,lr,ax,ax1)

#plot individual graphs
_,ax = plt.subplots(figsize=(14,10))
_,ax1 = plt.subplots(figsize=(14,10))
plot_GD(1000,0.01,ax,ax1)

#stochastic gradient descent
#start points are random

def SGD(X,y,theta,learning_rate=0.01,iterations=10):
	'''
	X = matrix X with added bias units
	y = vector of y
	theta = starting thetas
	'''
	m = len(y)
	cost_history = np.zeros(iterations)
	for itr in range(iterations):
		#init cost
		cost = 0.0
		#loop through predictions
		for i in range(m):
			#random number up to length y
			rand_ind = np.random.randint(0,m)
			x_i = X[rand_ind,:].reshape(1,X.shape[1])
			y_i = y[rand_ind,:].reshape(1,1)
			prediction = np.dot(x_i,theta)
			#adjust theta
			theta = theta - (1/m)*learning_rate*(x_i.T.dot((prediction-y_i)))
			#adjust the cost
			cost += cal_cost(theta,x_i,y_i)
		cost_history[itr] = cost
	return(theta,cost_history)

#test SGD function
lr = 0.6
n_iter = 50
theta = np.random.randn(2,1)
X_b = np.c_[np.ones((len(X),1)),X]
theta,cost_history = SGD(X_b,y,theta,lr,n_iter)

print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))

fig,ax=plt.subplots(figsize=(10,8))
ax.set_ylabel("{J(Theta)",rotation=30)
ax.set_xlabel('{Iterations}')
theta = np.random.rand(2,1)
_=ax.plot(range(n_iter),cost_history,'b-')


def minibatch_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10,batch_size=20):
	'''
	X = matrix with no added bias units
	y = response vector
	theta = weights
	returns fhe final theta vectors and array of cost history and number of iterations
	'''
	m = len(y)
	cost_history = np.zeros(iterations)
	#cast as int
	n_batches = int(m/batch_size)

	for itr in range(iterations):
		#initilize the cost
		cost = 0.0
		#generate random indcies
		indices = np.random.permutation(m)
		X =X[indices]
		y = y[indices]
		#indices by batch size
		for i in range(0,m,batch_size):
			#get x's
			X_i = X[i:i+batch_size]
			#get y's
			y_i = y[i:i+batch_size]
			#add bias
			X_i = np.c_[np.ones(len(X_i)),X_i]
			#predict
			prediction = np.dot(X_i,theta)
			#get the erros
			error = prediction - y_i
			#update theta
			theta = theta - (1/m)*learning_rate*(X_i.T.dot((error)))
			#calculate the cost 
			cost += cal_cost(theta,X_i,y_i)
		#add cost to iteration of history
		cost_history[itr] = cost
	return(theta,cost_history)

#test out mini batch
lr = 0.01
n_iter = 1000
theta = np.random.randn(2,1)

theta,cost_history = minibatch_gradient_descent(X,y,theta,lr,n_iter)
theta[0][0]
theta[1][1]
cost_history[-1]

#plot cost function against iterations and see how quickly it decreases
fig,ax = plt.subplots(figsize=(10,8))

ax.set_ylabel('{J(Theta)}',rotation=0)
ax.set_xlabel('{Iterations}')
theta = np.random.randn(2,1)

_=ax.plot(range(n_iter),cost_history,'b.')











































































