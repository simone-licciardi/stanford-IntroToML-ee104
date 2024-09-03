import numpy as np

def RMS(predicted,target):
        """This function computes the RMS, given the predicted data and target data.
        
        INPUT
        =====
        
        OUTPUT
        ======
        
        MODEL
        ====="""
        if np.size(np.shape(predicted)) == 1:
                return np.sqrt(np.mean((predicted-target)**2))
        else:
                return np.sqrt(np.mean(np.norm(predicted-target,2,axis=1)))