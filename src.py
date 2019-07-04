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
 1. Particles are hard pheres, and thus not allowed to intersect in any way, 
 e.g. their distance should be greater than the diameter d of the particles
 2. Particles are moving within a box with periodic boundary conditions.
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
 ''' Initial configuration uniformly distributed within dim-dimensional Box'''
 a = np.zeros((NParticles,dim))
 for i in range(NParticles):
  for j in range(dim):
   a[i,j] = np.random.uniform(Box[j,0],Box[j,1])
 return a

def Sampler(N,NParticles,d,norm,Box,bx,dim):
 ''' MCMC real space position sampler of hard spheres in dim-dimensional Box. 
 Essentially based on Krauth's hard spheres MCMC method''' 
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

def ConnectedDist(R,r_c,dim): 
 '''Tuples of indices of directly connected particles
 E.g. (1,2) means that particle 1 and particle 2 are directly connected since their minimal distance
 is smaller than the critical distance r_c.'''
 n = len(R)
 g = []
 D = np.zeros((n,n))
 for i in range(n):
  ri = np.array(R[i,:])
  for j in range(n):
   rj = np.array(R[j,:])
   D[i,j] = np.sqrt(Scalar(ri-rj,ri-rj,dim))
   if D[i,j] <= r_c and i!=j:
    g += [(i,j)]
 g = set(g)
 return list(set((a,b) if a<=b else (b,a) for a,b in g)),D

def ConnectedTuples(pairs):
 '''Merge Tuples having at least one item in common.
 E.g. (1,2),(1,4) -> (1,2,4) means since particle 1 is connected to particle 2 and particle 4, 
 particles 1,2,4 form a cluster'''
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

def ClusterLength(R,r_c,dim):
 '''Identification of directly connected particles, clusters, largest cluster and size of the largest cluster.'''
 k = ConnectedDist(R,r_c,dim)
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
 return C,g,max(g),C[indmaxf[0]]
