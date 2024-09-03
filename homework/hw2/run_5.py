import json
import random as rd

import numpy as np
import matplotlib.pyplot as plt

from ee104.hw2 import SoftNN, RMS

def import_json():
        with open('feat_valid.json','r') as file:
                data = json.load(file)
        U = np.array(data['U']['data'])
        v = np.array(data['v']['data'])
        return (U,v)

def partition_data(U,v,p): # p is the proportion of data devoted to test set
        [N,m] = np.shape(U)
        index = np.array(range(N))
        rd.shuffle(index)
        return(U[index > N*p],U[index <= N*p],v[index > N*p],v[index <= N*p])

def feature_map_1(U):
        return U
def feature_map_2(U):
        return np.concatenate([U,np.stack([U[:,0]*U[:,1],U[:,2]*U[:,1],U[:,2]*U[:,0]]).T,U**2],axis=1)

[U,v] = import_json()

[U, Ut, v, vt] = partition_data(U,v,.2)
[X1, X1t] = [feature_map_1(U),feature_map_1(Ut)]
[X2, X2t] = [feature_map_2(U),feature_map_2(Ut)]

R = np.linspace(.1,1.5,20);

RMStrain1 = np.empty(np.shape(R)); RMStest1 = np.empty(np.shape(R))
RMStrain2 = np.empty(np.shape(R)); RMStest2 = np.empty(np.shape(R))
for i in range(np.size(R)):
        RMStrain1[i] = RMS(SoftNN(X1,X1,v,R[i]),v)
        RMStest1[i] = RMS(SoftNN(X1t,X1,v,R[i]),vt)
        RMStrain2[i] = RMS(SoftNN(X2,X2,v,R[i]),v)
        RMStest2[i] = RMS(SoftNN(X2t,X2,v,R[i]),vt)

plt.plot(R,RMStrain1,color="blue",label="train - 1 feature map")
plt.plot(R,RMStest1,color="red",label="test - 1 feature map")
plt.plot(R,RMStrain2,color="green",label="train - 2 feature map")
plt.plot(R,RMStest2,color="purple",label="test - 2 feature map")
plt.legend()
plt.show()

# the result makes sense since the scale of the problem is 1, indeed.
# so, the second and first feature maps, with smoother parameters are best.

# at start strong underfit given by nearly piecewise approximation.
# at the end, it is nearly constant and stays so.