#------Real Space Configuration plotter
#------Shows graphs of connected particles
from src import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera
import matplotlib.cm as cm
from collections import OrderedDict
cmaps = OrderedDict()
b = Box([1,1],2)
bx = b[:,1]
Rt = np.genfromtxt("rsdis.txt",delimiter=",")
Rx = np.array([[Rt[i,0]+2*bx[0],Rt[i,1]] for i in range(len(Rt))])
Ry = np.array([[Rt[i,0],Rt[i,1]+2*bx[1]] for i in range(len(Rt))])
Rxx = np.array([[Rt[i,0]-2*bx[0],Rt[i,1]] for i in range(len(Rt))])
Ryy = np.array([[Rt[i,0],Rt[i,1]-2*bx[1]] for i in range(len(Rt))])
R = np.concatenate((Rt,Rx,Ry,Rxx,Ryy),axis=0)
R = np.delete(R,R[:,0]<(b[0,0]-0.1),axis=0) 
R = np.delete(R,R[:,1]<(b[1,0]-0.1),axis=0)
R = np.delete(R,R[:,0]>(b[0,0]+0.1),axis=0)
R = np.delete(R,R[:,1]>(b[1,0]+0.1),axis=0)

r_c = 0.4 
dim = 2
Cl = ClusterLength(R,r_c,dim,bx)

plt.xlim(b[0,0],b[0,1])
plt.ylim(b[1,0],b[1,1])

plt.scatter(R[:,0],R[:,1],marker="o",c="k",s=5)

colors = cm.plasma(np.linspace(0, 1, len(Cl[3])))

for i,c in zip(range(len(Cl[3])),colors):
 plt.scatter(R[Cl[3][i],0],R[Cl[3][i],1],marker="o",s=5,color=c)
#for j in Cl[3]:
# for k in Cl[3]:
#  plt.plot([R[j,0],R[k,0]],[R[j,1],R[k,1]],alpha=0.3,color="r")

cl0 = Cl[0]
print(Cl) 
def col_cycler(cols):
 count = 0
 while True:
  yield cols[count]
  count = (count + 1)%len(cols)
collor = col_cycler(['g', 'purple', 'b', 'gray', 'orange', 'r','c', 'm', 'y', 'k'])


for i in range(len(cl0)):
 cl0i = list(cl0[i])
 cmaa = next(collor)
 for j in range(len(cl0i)-1):
  plt.plot([R[cl0i[j],0],R[cl0i[j+1],0]],[R[cl0i[j],1],R[cl0i[j+1],1]],c=cmaa,alpha=0.1)
plt.show()
plt.savefig("rsdis.eps")
