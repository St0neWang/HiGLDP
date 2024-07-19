import numpy as np
from DAE import DAE



lnc = np.loadtxt(r"../feature/Lnc_vector.txt")
do = np.loadtxt(r"../feature/DO_vector.txt")

lnc_size=lnc.shape[1]
do_size=do.shape[1]


data1=DAE(lnc,lnc_size,20,16,1,600,[600])
np.savetxt('../Lnc_dim600.txt',data1)

data2=DAE(do,do_size,20,32,1,400,[400])
np.savetxt('../DO_dim400.txt',data2)