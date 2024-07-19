import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos


def construct_graph(features, topk):
    fname = '../knn/tmp.txt'
    f = open(fname, 'w')
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_knn():

        topk = 1
        data = np.loadtxt('LDA_feature.txt', dtype=float)
        construct_graph(data, topk)
        f1 = open('../knn/tmp.txt', 'r')
        f2 = open('../knn/c'+str(topk)+'.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{}\t{}\n'.format(start, end))
        f2.close()

generate_knn()
