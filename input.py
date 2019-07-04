from src import *

#----------------------------Initialization------------------------------------

b = Box([4,4],2)
bx = b[:,1]
dim = 2
N=10
NPar=100
d=0.5
norm=0.2
r_c=0.4

#------------------------------Simulation--------------------------------------

t0 = time.time()
R=Sampler(N,NPar,d,norm,b,bx,dim)
t1 = time.time()-t0

Cl = ClusterLength(R,r_c,dim)

f = open("rsdis.txt","w")
for i in range(len(R)):
 f.write("%s, %s \n"%(R[i,0],R[i,1]))
f.close()
g = open("test_data.txt","a")
g.write("%s,%s \n"%(NPar,Cl[2]))
g.close()
