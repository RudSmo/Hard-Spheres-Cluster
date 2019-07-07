import numpy as np

pcl = np.genfromtxt("pclt_data.txt",delimiter=",")[:,1]
l = np.genfromtxt("clstr_data.txt",delimiter=",")[:,1]
acc = np.genfromtxt("acc_rate.txt",delimiter=",")



pcl_re = np.genfromtxt("pclt_data_re.txt",delimiter=",")[:,1]
l_re = np.genfromtxt("clstr_data_re.txt",delimiter=",")[:,1]
acc_re = np.genfromtxt("acc_rate_re.txt",delimiter=",")

NPar=10
pcl_S = np.std(pcl)
pcl_M = np.average(pcl)

pcl_re_S = np.std(pcl_re)
pcl_re_M = np.average(pcl_re)

l_S = np.std(l)
l_M = np.average(l)
l_re_S = np.std(l_re)
l_re_M = np.average(l_re)


acc_S = np.std(acc)
acc_M = np.average(acc)

acc_re_S = np.std(acc_re)
acc_re_M = np.average(acc_re)


f = open("pcl_avg.txt","w")
f.write("%s,%s,%s"%(NPar,pcl_M,pcl_S))

f1 = open("pcl_re_avg.txt","w")
f1.write("%s,%s,%s"%(NPar,pcl_re_M,pcl_re_S))


g = open("clstr_avg.txt","w")
g.write("%s,%s,%s"%(NPar,l_M,l_S))
g1 = open("clstr_re_avg.txt","w")
g1.write("%s,%s,%s"%(NPar,l_re_M,l_re_S))

h = open("acc_avg.txt","w")
h.write("%s,%s,%s"%(NPar,acc_M,acc_S))

h1 = open("acc_re_avg.txt","w")
h1.write("%s,%s,%s"%(NPar,acc_re_M,acc_re_S))


