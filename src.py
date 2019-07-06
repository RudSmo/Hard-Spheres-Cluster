import numpy as np
import time
import matplotlib.animation as animation                    
import time                                                 
from celluloid import Camera 


def Box(r,dim):
 """dim-d Box parameters; defined just for convinience"""
 return np.array([[-r[i]/2,r[i]/2] for i in range(dim)])

def Scalar(x,y,dim):
 """Scalar product of dim-darrays x,y with equal dimensions dim"""
 return (np.array([x[i]*y[i] for i in range(dim)])).sum()

def mDist(r1,r2,bx):
 """ Minimal distance of two Box-darrays imposing periodic boundary condition.
  b = Box[:,1]. Better defined outside of big loops for performance issues"""
 dv = np.abs(r1-r2)
 dv = np.where(dv>bx,dv-2*bx,dv)
 return np.sqrt((dv**2).sum(axis=-1))

def AcceptableMove(r,R,d,norm,Box,bx,dim):
 ''' Moves Particles to allowed space. Allowed means here:
 1. Particles are not allowed to intersect, e.g. their distance should be greater than their diameter d
 2. Particles are moving within a box with pbc.
 ''' 
 ra = np.array(r)
 rr = norm*(np.random.rand(dim)-0.5)
 rn = ra+rr
 r_new = np.zeros(dim)
 for i in range(len(R)):
  dist = mDist(r,R[i],bx)
  for j in range(dim):
   if dist < d:
    r_new[j] = ra[j]
   else:
    r_new[j] = rn[j]
   if r_new[j] > Box[j,1]:
    r_new[j] = r_new[j] - bx[j]
   elif r_new[j] < Box[j,0]:
    r_new[j] = r_new[j] + bx[j]
   else:
    r_new[j] = r_new[j]
 return r_new 
    
def InitConfigRan(NParticles,Box,dim):
 a = np.zeros((NParticles,dim))
 for i in range(NParticles):
  for j in range(dim):
   a[i,j] = np.random.uniform(Box[j,0],Box[j,1])
 return a

def Sampler(N,NParticles,d,norm,Box,bx,dim):
#Implement Thermalization steps, number of steps before update, IO in general
#Implement Acceptance rate. Optimum: 0.23
 R0 = InitConfigRan(NParticles,Box,dim)
 #fig = plt.figure()
 #camera = Camera(fig)
 for i in range(N):
  if i == 0:
   Rc = R0
  else:
   Rc = Rc
  for j in range(NParticles):
   Rc[j] = AcceptableMove(Rc[j],Rc,d,norm,Box,bx,dim)
  #plt.scatter(Rc[:,0],Rc[:,1],marker="o",s=d,c="k")
  #camera.snap()
  #ani = camera.animate(blit=True)
  #ani.save('animation.gif', writer='imagemagick', fps=25)
 return Rc 

def ConnectedDist(R,r_c,dim,bx):
 n = len(R)
 g = []
 D = np.zeros((n,n))
 for i in range(n):
  ri = np.array(R[i,:])
  for j in range(n):
   rj = np.array(R[j,:])
   D[i,j] = mDist(ri,rj,bx)
   if D[i,j] <= r_c and i!=j:
    g += [(i,j)]
 g = set(g)
 return list(set((a,b) if a<=b else (b,a) for a,b in g)),D

def ConnectedTuples(pairs):
 L = {}
 def NewList(x, y):
  L[x] = L[y] = [x, y]
 def AddToList(lst, el):
  lst.append(el)
  L[el] = lst
 def MergeLists(lst1, lst2):
  mlist = lst1 + lst2
  for el in mlist:
   L[el] = mlist
 for x, y in pairs:
  xList = L.get(x)
  yList = L.get(y)
  if not xList and not yList:
   NewList(x, y)
  if xList and not yList:
   AddToList(xList, y)
  if yList and not xList:
   AddToList(yList, x)
  if xList and yList and xList != yList:
   MergeLists(xList, yList)
 return list(set(tuple(l) for l in L.values()))

def ClusterLength(R,r_c,dim,bx):
 k = ConnectedDist(R,r_c,dim,bx)
 C = ConnectedTuples(k[0])
 D = k[1]
 A = []
 indmax = []
 g = np.zeros(len(C))
 for i in range(len(C)):
  for j in C[i]:
   for l in C[i]:
    A.append(D[j,l])
    g[i] = max(A)
    indmax.append([m for m, h in enumerate(A) if h == g[i]])
 indmaxf = [m for m, h in enumerate(g) if h == max(g)]
 if len(C)==0:
  return 0.0,0.0,0.0,0.0
 else:
  return C,g,max(g),C[indmaxf[0]]

def Percolate(R,r_c,dim,bx):
 bx1 = 3*bx
 p = 0
 Cl = ClusterLength(R,r_c,dim,bx)
 if dim == 2:
  Rx = np.array([[R[i,0]+2*bx[0],R[i,1]] for i in range(len(R))])
  Ry = np.array([[R[i,0],R[i,1]+2*bx[1]] for i in range(len(R))])
  Rxx = np.array([[R[i,0]-2*bx[0],R[i,1]] for i in range(len(R))])
  Ryy = np.array([[R[i,0],R[i,1]-2*bx[1]] for i in range(len(R))])
  R1 = np.concatenate((R,Rx,Ry,Rxx,Ryy),axis=0)
  Cl1 = ClusterLength(R1,r_c,dim,bx1)
  if Cl1[0] == 0:
   p = 0
  elif len(Cl1[0]) == len(Cl[0])*2**(dim)+1:
   p = 0
  else:
   p = 1
 else:  
  raise ValueError("Other Dimensions not implemented yet. Only dim=2 works.")
 return p

#----------------------------Initialization------------------------------------

b = Box([3,3],2)
bx = b[:,1]
dim = 2
N=100
NPar=1
d=0.5
norm=0.2
r_c=0.4
#------------------------------Simulation--------------------------------------
t0 = time.time()
R=Sampler(N,NPar,d,norm,b,bx,dim)
t1 = time.time()-t0

Cl = ClusterLength(R,r_c,dim,bx)
Pc = Percolate(R,r_c,dim,bx)
f = open("rsdis.txt","w")
for i in range(len(R)):
 f.write("%s, %s \n"%(R[i,0],R[i,1]))
f.close()
g = open("clstr_data.txt","a")
g.write("%s,%s \n"%(NPar,Cl[2]))
g.close()
h = open("pclt_data.txt","a")
h.write("%s,%s \n"%(NPar,Pc))
h.close()
