import numpy as np

def SoftNN_w(x,features,scale):
        """This function computes weights for the Soft Nearest Neighbor algorithm, with scale parameter p.
        
        INPUT
        =====
        x : a matrix of datapoints (like the validation set or test set), formatted as NxD matrix of N D-dimensional (feature-mapped) points.
        features : NxD matrix of N D-dimensional features.
        labels: Nx1 matrix of N 1-dimensional labels.
        scale: the best result is obtained when the parameter (below, p) here is similar to the scale of the problem over the dominion (that is, in the same order of magnitude of distances between x's).
        
        OUTPUT
        ======
        the vector of weights
        
        MODEL
        =====
        We consider features = (x^i)_i=0^N-1 and labels = (y^i)_i=0^N-1. The mathematical model for the weights is, componentwise, given by
        
        w_i(x) = \frac{\exp(-||x-x^i||_2^2/p^2)}{\sum_j=0^N-1 \exp(-||x-x^j||_2^2/p^2)}."""
        
        E = np.exp(- 1/(scale**2) * np.linalg.norm(x-features,2,axis=1))
        return E / np.sum(E) 
    
    
def SoftNN(x,features,labels,scale):
        """This function computes the Soft Nearest Neighbor algorithm, with scale parameter p.
        
        INPUT
        =====
        x : a matrix of datapoints (like the validation set or test set), formatted as NxD matrix of N D-dimensional (feature-mapped) points.
        features : NxD matrix of N D-dimensional features.
        labels: Nx1 matrix of N 1-dimensional labels.
        scale: the best result is obtained when the parameter (below, p) here is similar to the scale of the problem over the dominion (that is, in the same order of magnitude of distances between x's).
        
        OUTPUT
        ======
        the vector of weights
        
        MODEL
        =====
        We consider features = (x^i)_i=0^N-1 and labels = (y^i)_i=0^N-1. The mathematical model is
        
        \cap{y} = \sum^N-1_i=0 w_i(x) y^i
        
        where the weights are, componentwise, given by
        
        w_i(x) = \frac{\exp(-||x-x^i||_2^2/p^2)}{\sum_j=0^N-1 \exp(-||x-x^j||_2^2/p^2)}."""  

        # Here one could use a for and the function SoftNN_w.
        
        # n = np.size(x,axis = 0)
        # res = np.empty(n)
        # for i in range(n):
        #         w = SoftNN_w(x[i],features,scale)
        #         res[i] = np.dot(w,labels)
        # return res      
        
        # A more elegant solution is proposed, using Broadcasting from NumPy
        
        x = x[np.newaxis,:,:]
        features = features[:,np.newaxis,:]
        E = np.exp(- 1/(scale**2) * np.linalg.norm(x-features,2,axis=2)) # Broadcasting => x-features[i,j,:]=x^i-features_j
        W = E / (np.sum(E,axis=1)[:,np.newaxis])
        return W.T @ labels