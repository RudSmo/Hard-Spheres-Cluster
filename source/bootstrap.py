import numpy as np
import pandas as pd


def bootstrap_resample(X, n=None):
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample

size=50000
pclt = np.genfromtxt("pclt_data.txt",delimiter=",")[:,1]
pclt_re = bootstrap_resample(pclt,size)

l = np.genfromtxt("clstr_data.txt",delimiter=",")[:,1]
l_re = bootstrap_resample(l,size)
acc = np.genfromtxt("acc_rate.txt",delimiter=",")
acc_re = bootstrap_resample(acc,size)


f = open("pclt_data_re.txt","w")
for i in range(size):
 f.write(",%s\n"%(pclt_re[i]))
f.close()
g = open("acc_rate_re.txt","w")
for i in range(size):
 g.write(",%s\n"%(acc_re[i]))
g.close()
h = open("clstr_data_re.txt","w")
for i in range(size):
 h.write(",%s\n"%(l_re[i]))
h.close()

