import numpy as np
import pandas as pd


def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample

size=50000
X = np.genfromtxt("test_data.txt",delimiter=",")[:,1]
X_re = bootstrap_resample(X,size)
f = open("test_data_re.txt","w")
for i in range(size):
 f.write(",%s\n"%(X_re[i]))
f.close()
