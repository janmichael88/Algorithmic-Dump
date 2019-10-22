import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

x = np.linspace(-1.5,1.5)
[X,Y] = np.meshgrid(x,x)

#fig = plt.figure()
#ax = fig.gca(projection = '3d')
#plot the surface
#ax.plot_surface(X,Y,X+Y)

#plot the constraint
theta = np.linspace(0,2*np.pi)
R = 1.0
x1 = np.cos(theta)
y1 = np.sin(theta)

#ax.plot(x1,y1,x1+y1,'r-')
#plt.tight_layout()

#function to maxize os
#f(x,y) = x + y
#constraint is g(x,y) = x^2 + y^2 -1
#langrangian would be L(x,y|l) = f(x,y) + l*g(x,y)

#construct the lagrange multiplied augmented function
def lagrangian(X):
	#X is a vector
	x = X[0]
	y = X[1]
	L =X[2]
	return(x + y + L*(x**2 + y**2 -1))

#define diff lagrangian, using finite differences
def diff_lagrangian(X):
	#initialize
	diff_Lambda = np.zeros(len(X))
	#define the step size
	h = 1e-3
	for i in range(len(X)):
		dX = np.zeros(len(X))
		dX[i] = h
		diff_Lambda[i] = (lagrangian(X+dX) - lagrangian(X-dX)) / (2*h)
	return(diff_Lambda)

#find the max or min
#setting the derivative of dfunc to zero

#get the max
X_max = fsolve(diff_lagrangian,[1,1,0])
print(X_max,lagrangian(X_max))
#X_max are the values at the man, lagrangian gives the actual max

X_min = fsolve(diff_lagrangian,[-1,-1,0])
print(X_min,lagrangian(X_min))

#plotting the space
#fig2 = plt.figure() #create an empty figure with no axes
#ax2 = Axes3D(fig2) #created a 3d axes object

#plotting the surface
#fig3 = plt.figure()
#ax3 = Axes3D(fig3)
#ax3.plot_surface(X, Y, X+Y, color='y', rstride=1, cstride=1, alpha=0.5, edgecolor='w')
#plt.show()

#add in the constraint
#fig4 = plt.figure()
#ax4 = Axes3D(fig4)
#ax4.plot_surface(X, Y, X+Y, color='y', rstride=1, cstride=1, alpha=0.5, edgecolor='w')

#theta = np.linspace(0, 2*np.pi)
#R = 1.0
#x1 = R * np.cos(theta)
#y1 = R * np.sin(theta)
#ax4.plot(x1, y1, x1+y1)


#add in the points that optimize the function subject to the constraint
fig5 = plt.figure()
ax5 = Axes3D(fig5)
ax5.plot_surface(X, Y, X+Y, color='y', rstride=100, cstride=100, alpha=0.2, edgecolor='w')
#ax5.plot_surface(X, Y, 0, color='b', alpha=0.05, rstride=100, cstride=100)

theta = np.linspace(0, 2*np.pi)
R = 1.0
x1 = R * np.cos(theta)
y1 = R * np.sin(theta)
ax5.plot(x1, y1, alpha=0.3)
ax5.plot(x1, y1, x1+y1, c='y', alpha=0.8)

ax5.scatter(X_max[0], X_max[1], X_max[2], s=30, c='r')
ax5.scatter(X_min[0], X_min[1], X_min[2], s=30, c='g')
plt.show()















