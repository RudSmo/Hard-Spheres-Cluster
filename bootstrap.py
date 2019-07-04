import numpy as np
import pandas as pd


def resample(X,n):
    if n == None:
        n = len(X)
    res = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_r = X[res]
    return X_r

size=50000

X = np.genfromtxt("test_data.txt",delimiter=",")[:,1]
X_r = resample(X,size)
f = open("test_data_re.txt","w")
for i in range(size):
 f.write(",%s\n"%(X_re[i]))
f.close()
