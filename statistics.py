import numpy as np

X = np.genfromtxt("pclt_data_re.txt",delimiter=",")[:,1]

NPar=10
S = np.std(X)
M = np.average(X)

f = open("avg.txt","w")
f.write("%s,%s,%s"%(NPar,M,S))
