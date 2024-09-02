import numpy as np
import matplotlib.pyplot as plt

from ee104.hw1 import SoftNN, kNN

# algorithms

# def KNN(xd, xv, yv, n):

# problem parameters

def f(x):
        if x >= 0 and x <= 1:
                return np.sin(10*x)
f_np = np.vectorize(f)
        
N = 30;

# random generated data

xvec = np.random.rand(N,1)
yvec = f_np(xvec)

# visualization and comparison

xview = np.linspace(0, 1, 500)

# over-same-graph comparison

# plt.scatter(xvec,yvec,label='random data')
# plt.plot(xview,f_np(xview),label='sin(10x)',color='red')
# plt.plot(xview,SoftNN(xview, xvec, yvec, .2),label='snn',color='green')
# plt.plot(xview,kNN(xview, xvec, yvec,3),label='knn',color='purple')
# plt.legend()
# plt.title("exercize 3a")
# plt.show()

# different-subplots + numerical comparison
plt.subplot(2,5,1)
yview = f_np(xview)
plt.scatter(xvec,yvec,label='random data')
plt.plot(xview,yview,label='knn',color='purple')
        
E = np.empty(8);
for k in [1,2,3]:
        plt.subplot(2,5,k+2)
        currview = kNN(xview, xvec, yvec,k)
        plt.scatter(xvec,yvec,label='random data')
        plt.plot(xview,currview,label='knn',color='purple')
        plt.title("k-nearest neighbor")
        E[k-1] = np.linalg.norm(yview-currview,2)/np.power(500,.5)
 
P = np.array([.5,1,3,5,10]) * np.sqrt(.001)
for k in range(5):
        plt.subplot(2,5,k+6)
        currview = SoftNN(xview, xvec, yvec,P[k])
        plt.scatter(xvec,yvec,label='random data')
        plt.plot(xview,currview,label='snn',color='red')
        plt.title("p soft nearest neighbor")
        E[3+k] = np.linalg.norm(yview-currview,2)/np.power(500,.5)

print(E)
plt.show()
