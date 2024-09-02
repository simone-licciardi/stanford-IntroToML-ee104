import numpy as np

def SNN_weights(x,xv,p):
        n = np.size(xv, axis = 0) # counts rows
        diff = np.tile(x,(n,1)) # matrix with n observations of xd
        diff = diff - xv # they have the same dim by construction
        diff = np.linalg.norm(diff,2,axis = 1)
        exps = np.exp(- diff / pow(p,2))
        return exps / np.sum(exps)
        
def SoftNN(xd, xv, yv, p):
        "xv is a n x d vector, with d sample dimension, and n sample dataset cardinality"
        n = np.size(xd,axis = 0)
        res = np.empty(n)
        for i in range(n):
                w = SNN_weights(xd[i],xv,p)
                res[i] = np.dot(w,yv)
        return res