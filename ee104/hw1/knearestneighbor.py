import numpy as np

def kNN_near(x,xv,k):
	n = np.size(xv, axis = 0) # counts rows
	diff = np.tile(x,(n,1)) # matrix with n observations of xd
	diff = diff - xv # they have the same dim by construction
	diff = np.linalg.norm(diff,2,axis = 1)
	return np.argpartition(diff, k)[:k]
        
def kNN(xd, xv, yv, k):
	"xv is a n x d vector, with d sample dimension, and n sample dataset cardinality"
	N = np.size(xd,axis = 0)
	res = np.empty(N)
	for i in range(N):
		nearest = kNN_near(xd[i],xv,k)
		res[i] = np.sum(yv[nearest])/np.size(nearest)
	return res
