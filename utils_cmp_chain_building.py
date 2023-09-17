import numpy as np
import faiss
from utils import build_chain_org, build_chain_alt, build_chain_new


def cmp_chain_building():
    # test/compare chain building implementations

    k = 20
    np.random.seed(seed=6)
    x = np.random.random((7500, 100)).astype(np.float32)

    # # nearest neighbour object
    index = faiss.IndexFlatL2(x.shape[-1])
    neighbor_mask = np.ones(len(x))
    # # add nearest neighbour candidates
    index.add(x[neighbor_mask == 1])

    # # distances and idx of neighbour points for the neighbour candidates (k+1 as the first one will be the point itself)
    dist_train, idx_train = index.search(x[neighbor_mask == 1], k=k+1)

    # remove 1st nearest neighbours to remove self loops
    dist_train, idx_train = dist_train[:, 1:], idx_train[:, 1:]  # no duplicates, so we can use this simplidied version

    dist, idx = dist_train, idx_train

    pts_all = x[neighbor_mask == 1]

    from tqdm import tqdm  # optional prograss bar (since this part takes a while)
    cnt_same_length = 0
    for i in tqdm(range(len(idx))):
        for ii in range(len(idx[i])):
            chain_length = ii + 1

            chain_distances_new, used_new = build_chain_new(pts_all, idx, i, chain_length)

            chain_distances_alt, used_alt = build_chain_alt(dist, idx, i, chain_length)
            # chain_distances_org, used_org = build_chain_org(dist, idx, i, chain_length, k)

            if len(used_alt) == len(used_new):
                # if alt method found full chain, results are equal, except when there are duplicates/rounding errors
                assert used_new == used_alt
                assert np.allclose(chain_distances_new, chain_distances_alt, atol=1.e-6)
                cnt_same_length += 1
    print('confirmed chains of same length', cnt_same_length)


if __name__ == "__main__":
    cmp_chain_building()
