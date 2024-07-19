import numpy as np
from tqdm import tqdm
    
lnc = np.loadtxt(r".\Lnc_dim600.txt", dtype=float)
dis = np.loadtxt(r".\DO_dim400.txt", dtype=float)


def write_feature():
    index = np.loadtxt(r".\data\index.txt", dtype=int)
    new_feature = np.zeros((22622, 1000))
    for i in tqdm(range(22622)):
        new_feature[i, :600] = lnc[index[i, 1]]
        new_feature[i, 600:] = dis[index[i, 2]]
    np.savetxt(r"LDA_feature.txt", new_feature, fmt='%.15f')

write_feature()