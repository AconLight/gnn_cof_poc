import numpy as np
import faiss
from utils import build_chain_org, build_chain_alt


def cmp_chain_building():
    # test/compare chain building implementations

    k = 5
    np.random.seed(seed=6)
    x = np.random.random((10, 2))  # random, but interesting data

    # # nearest neighbour object
    index = faiss.IndexFlatL2(x.shape[-1])
    neighbor_mask = np.ones(len(x))
    # # add nearest neighbour candidates
    index.add(x[neighbor_mask==1])

    # # distances and idx of neighbour points for the neighbour candidates (k+1 as the first one will be the point itself)
    dist_train, idx_train = index.search(x[neighbor_mask==1], k = k+1)

    # remove 1st nearest neighbours to remove self loops
    dist_train, idx_train = dist_train[:,1:], idx_train[:,1:]  # no duplicates, so we can use this simplidied version

    dist, idx = dist_train, idx_train
    
    # print('data points', x)
    print('dist', dist)
    print('idx = neighbours in order by distance')
    for i, nbrs in enumerate(idx):
        print(i, nbrs)
    print()

    for i in range(2): # len(idx)):
        chain_length = len(idx[i])  # we need long chain
        print(f'\nstart node={i}, chain_length={chain_length}')

        print('org implementation')
        chain_distances, used = build_chain_org(dist, idx, i, chain_length, k=k)
        print('used:', used)
        print('chain_distances:', chain_distances)

        print('alt implementation')
        chain_distances, used = build_chain_alt(dist, idx, i, chain_length)
        print('used:', used)  # it has len+1 vs distances, bc there are more nodes than edges on path
        print('chain_distances:', chain_distances)


if __name__ == "__main__":
    cmp_chain_building()
