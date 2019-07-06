import numpy as np
import pandas as pd


def bootstrap_resample(X, n=None):
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample

size=50000
X = np.genfromtxt("pclt_data.txt",delimiter=",")[:,1]
X_re = bootstrap_resample(X,size)
f = open("pclt_data_re.txt","w")
for i in range(size):
 f.write(",%s\n"%(X_re[i]))
f.close()
