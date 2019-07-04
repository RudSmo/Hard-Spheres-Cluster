import numpy as np

X = np.genfromtxt("test_data.txt",delimiter=",")[:,1]

NPar = 10
S = np.std(X)
M = np.average(X)

f = open("avg.txt","w")
f.write("%s,%s,%s"%(NPar,M,S))
