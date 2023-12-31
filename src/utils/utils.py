import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from pyod.models.abod import ABOD
import torch
from torch_geometric.data import Data
from scipy.io import loadmat
import faiss

########################################### NEGATIVE SAMPLE FUNCTIONS################################################
def negative_samples(normal_x, normal_y, anom_x, anom_y, k, sample_type, proportion, epsilon):

    test_idx = np.random.choice(np.arange(0, len(normal_x)), len(anom_x), replace=False)
    train_idx = np.setdiff1d(np.arange(0, len(normal_x)), test_idx)
    train_x = normal_x[train_idx]
    train_y = normal_y[train_idx]
    test_x = np.concatenate((normal_x[test_idx], anom_x))
    test_y = np.concatenate((normal_y[test_idx], anom_y))


    train_x, train_y, val_x, val_y, test_x, test_y = split_data(1, all_train_x=train_x, all_train_y=train_y,
                                                                all_test_x=test_x, all_test_y=test_y)

    # training set negative samples
    neg_train_x, neg_train_y = generate_negative_samples(train_x, sample_type, proportion, epsilon)
    # validation set negative samples
    neg_val_x, neg_val_y = generate_negative_samples(val_x, sample_type, proportion, epsilon)

    # concat data
    x = np.vstack((train_x, neg_train_x, val_x, neg_val_x, test_x))
    y = np.hstack((train_y, neg_train_y, val_y, neg_val_y, test_y))

    # all training set
    train_mask = np.hstack((np.ones(len(train_x)), np.ones(len(neg_train_x)),
                            np.zeros(len(val_x)), np.zeros(len(neg_val_x)),
                            np.zeros(len(test_x))))
    # all validation set
    val_mask = np.hstack((np.zeros(len(train_x)), np.zeros(len(neg_train_x)),
                          np.ones(len(val_x)), np.ones(len(neg_val_x)),
                          np.zeros(len(test_x))))
    # all test set
    test_mask = np.hstack((np.zeros(len(train_x)), np.zeros(len(neg_train_x)),
                           np.zeros(len(val_x)), np.zeros(len(neg_val_x)),
                           np.ones(len(test_x))))
    # normal training points
    neighbor_mask = np.hstack((np.ones(len(train_x)), np.zeros(len(neg_train_x)),
                               np.zeros(len(val_y)), np.zeros(len(neg_val_x)),
                               np.zeros(len(test_y))))
    
    # find k nearest neighbours (idx) and their distances (dist) to each points in x within neighbour_mask==1
    idx, dist, dist_cof, dist_vectors = find_neighbors(x, y, neighbor_mask, k)

    return x.astype('float32'), y.astype('float32'), neighbor_mask.astype('float32'), train_mask.astype('float32'), test_mask.astype('float32'), dist, dist_cof, dist_vectors, idx


# loading negative samples
def generate_negative_samples(x, sample_type, proportion, epsilon):
    n_samples = int(proportion * (len(x)))
    n_dim = x.shape[-1]

    # M
    randmat = np.random.rand(n_samples, n_dim) < 0.3
    # uniform samples
    rand_unif = (epsilon * (1 - 2 * np.random.rand(n_samples, n_dim)))
    #  subspace perturbation samples
    rand_sub = np.tile(x, (proportion, 1)) + randmat * (epsilon * np.random.randn(n_samples, n_dim))

    if sample_type == 'UNIFORM':
        neg_x = rand_unif
    if sample_type == 'SUBSPACE':
        neg_x = rand_sub
    if sample_type == 'MIXED':
        # randomly sample from uniform and gaussian negative samples
        neg_x = np.concatenate((rand_unif, rand_sub), 0)
        neg_x = neg_x[np.random.choice(np.arange(len(neg_x)), size=n_samples)]

    neg_y = np.ones(len(neg_x))

    return neg_x.astype('float32'), neg_y.astype('float32')

# loading negative samples
def generate_negative_samples_new(x, sample_type, proportion, epsilon):
    MIXEDHIGHDIM_FACTOR = 10
    if sample_type == 'MIXEDHIGHDIM':
        proportion *= MIXEDHIGHDIM_FACTOR

    n_samples = int(proportion*(len(x)))
    n_dim = x.shape[-1]
        
    #M
    randmat = np.random.rand(n_samples,n_dim) < 0.3
    # uniform samples
    rand_unif = (epsilon* (1-2*np.random.rand(n_samples,n_dim)))
    #  subspace perturbation samples
    # rand_sub = np.tile(x, (1,1)) + randmat*(epsilon*np.random.randn(n_samples,n_dim))
    
    if sample_type == 'UNIFORM':
        neg_x = rand_unif
    # if sample_type == 'SUBSPACE':
    #     neg_x = rand_sub
    if sample_type == 'MIXED':
        # randomly sample from uniform and gaussian negative samples
        # neg_x = np.concatenate((rand_unif, rand_sub),0)
        # neg_x = neg_x[np.random.choice(np.arange(len(neg_x)), size = n_samples)]
        neg_x = rand_unif
    # if sample_type == 'MIXEDHIGHDIM':
    #     # randomly sample from uniform and gaussian negative samples
    #     neg_x = np.concatenate((rand_unif, rand_sub),0)
    #     neg_x = neg_x[np.random.choice(np.arange(len(neg_x)), size = n_samples)]
    #     abod_model = ABOD(contamination=0.2, method='fast', n_neighbors=3)
    #     abod_model.fit(x)
    #
    #     abod_model = ABOD(contamination=0.2, method='fast', n_neighbors=3)
    #     abod_model.fit(np.concatenate((neg_x, x), 0))
    #     neg_x_scores = abod_model.decision_scores_
    #     neg_x_dim = [p for _, p in sorted(zip(neg_x_scores[:len(neg_x)], neg_x), key=lambda p: p[0], reverse=True)][:int(n_samples/MIXEDHIGHDIM_FACTOR*0.7)]
    #     neg_x_dim = np.array(neg_x_dim)
    #     both = np.concatenate((neg_x, neg_x_dim),0)
    #     both = both[np.random.choice(np.arange(len(neg_x)), size=int(n_samples/MIXEDHIGHDIM_FACTOR))]
    #     neg_x = np.array(both)
    #     print('neg_x size', len(neg_x))

    neg_y = np.ones(len(neg_x))
    
    return neg_x.astype('float32'), neg_y.astype('float32')


def build_chain_org(dist, idx, i, chain_length, k):
	# oryginal code extracted to method, so it can be compared / tested
    neighbor_chain = idx[i][0:chain_length]
    chain_distances = [dist[i][0]]
    used = [i]
    next_idx = idx[i][0]
    for iii in range(chain_length-1):
        ctr = 0
        while True:
            if ctr >= k:
                break
            found = idx[next_idx][ctr]
            if found in used or found not in neighbor_chain:
                ctr += 1
                continue
            chain_distances.append(dist[next_idx][ctr])
            used.append(found)
            next_idx = idx[found][0]
            break
    return chain_distances, used
    
    
def build_chain_alt(dist, idx, i, chain_length):
    neighbor_chain = set(idx[i][0:chain_length])
    chain_distances = []
    used = [i]
    next_idx = i
    for _ in range(chain_length):
        for found, found_dist in zip(idx[next_idx], dist[next_idx]):
            if found in used or found not in neighbor_chain or found < 0:
                continue
            chain_distances.append(found_dist)
            used.append(found)
            next_idx = found
            break
        else:
            # can't build long enough chain (with used number of nearest neighbours - full graph always allows to build chain)
            break

    return chain_distances, used


def pairwise_distnces_squared(x):
    # maybe there are better implementations
    x2 = np.einsum('ij,ij->i', x, x)
    return np.abs(x2[:, np.newaxis] + x2[np.newaxis, :] - 2 * x @ x.T)

    # probably more preciese
    # import sklearn.metrics.pairwise
    # return np.square(sklearn.metrics.pairwise_distances(x))


def build_chain_new(pts_all, idx, i, chain_length):
    # we dont use kNN distances here.
    # we use starting point kNN neighbours indices and calculate all pairwise distances
    # (because of that, its performance depends on problem dimentionality)

    neighbor_chain = [i] + idx[i][0:chain_length].tolist()

    pts = pts_all[neighbor_chain]

    pairwise_distnces_for_neighbours = pairwise_distnces_squared(pts)
    
    marker = np.inf  # way to block item from selection
    np.fill_diagonal(pairwise_distnces_for_neighbours, marker)

    next_idx = 0  # his local position (in pairwise_distnces_for_neighbours)
    used = [neighbor_chain[next_idx]]  # his global position
    chain_distances = []
    for _ in range(len(neighbor_chain)-1):
        pairwise_distnces_for_neighbours[:,next_idx] = marker
        found = np.argmin(pairwise_distnces_for_neighbours[next_idx])
        chain_distances.append(pairwise_distnces_for_neighbours[next_idx][found])
        used.append(neighbor_chain[found])
        next_idx = found

    # assert set(used) == set(neighbor_chain)  # we always find expected length chains
    return chain_distances, used
    

def calc_chain_value(chain_distances):
    chain_value = 0
    for c in range(len(chain_distances)):
        chain_value += (len(chain_distances)-c) / len(chain_distances) * chain_distances[c]
    chain_value /= len(chain_distances)
    return chain_value
    

################################### GRAPH FUNCTIONS ###############################################     
# find the k nearest neighbours of all x points out of the neighbour candidates
def find_neighbors(x, y, neighbor_mask, k):

    # nearest neighbour object
    index = faiss.IndexFlatL2(x.shape[-1])
    # add nearest neighbour candidates
    index.add(x[neighbor_mask==1])

    # distances and idx of neighbour points for the neighbour candidates (k+1 as the first one will be the point itself)
    dist_train, idx_train = index.search(x[neighbor_mask==1], k = k+1)
    # remove 1st nearest neighbours to remove self loops

    # dist_train, idx_train = dist_train[:,1:], idx_train[:,1:]
    # above line does not work when there are duplicate points in input data: self match can be @ idx > 0
    # 
    # sadly below code is not very readable - maybe there are better ways or even plain python would do
    # first lets get positions of self matches
    # (which may not be zero and in worse case (>k duplicates) could even be missing - causing problems for simpler approaches)
    self_positions_to_remove = np.argmax(idx_train == np.arange(idx_train.shape[0]).reshape(-1, 1), axis=1)
    idx_train[np.arange(len(self_positions_to_remove)), self_positions_to_remove] = -2  # mark positions to remove
    non_self_matches = idx_train != -2  # mask to use
    idx_train = idx_train[non_self_matches].reshape((-1, k))
    dist_train = dist_train[non_self_matches].reshape((-1, k))
    

    # distances and idx of neighbour points for the non-neighbour candidates
    dist_test, idx_test = index.search(x[neighbor_mask==0], k = k)
    #concat
    dist = np.vstack((dist_train, dist_test))
    idx = np.vstack((idx_train, idx_test))
    pts_all = np.vstack((x[neighbor_mask==1], x[neighbor_mask==0]))  # all points by index (needed for new approach)

    dist_vectors_shape = [idx.shape[0], idx.shape[1], x.shape[1]]
    dist_vectors = np.zeros(shape=dist_vectors_shape)


    for i in range(idx.shape[0]):
        for n_idx in range(idx.shape[1]):
            a = x[i]
            b = x[idx[i, n_idx]]
            c = a - b
            dist_vectors[i, n_idx] = c

    dist2 = dist.copy()
    # from tqdm import tqdm  # optional prograss bar (since this part takes a while)
    # for i in tqdm(range(len(idx))):
    #     # if i % int(len(idx)/100) == 0:
    #     #     print(str(int((i+1)/len(idx)*100)) + '%')
    #     for ii in range(len(idx[i])):
    #         chain_length = ii + 1
    #         # chain_distances, _ = build_chain_alt(dist, idx, i, chain_length)
    #         chain_distances, _ = build_chain_new(pts_all, idx, i, chain_length)
    #         dist2[i][ii] = calc_chain_value(chain_distances)


    return idx, dist, dist2, dist_vectors

def cut_data(dist, dist2, dist3,  idx, k):
    return dist[:, 0:k], dist2[:, 0:k], dist3[:, 0:k], idx[:, 0:k]



# create graph object out of x, y, distances and indices of neighbours
def build_graph(x, y, idx, distances_stack, is_vector=False):
    
    # array like [0,0,0,0,0,1,1,1,1,1,...,n,n,n,n,n] for k = 5 (i.e. edges sources)
    if not is_vector:
        idx_source = np.repeat(np.arange(len(x)),distances_stack[0].shape[-1]).astype('int32')
    else:
        idx_source = np.repeat(np.arange(len(x)),distances_stack[0].shape[-2]).astype('int32')
    idx_source = np.expand_dims(idx_source,axis=0)

    # edge targets, i.e. the nearest k neighbours of point 0, 1,..., n
    idx_target = idx.flatten()
    idx_target = np.expand_dims(idx_target,axis=0).astype('int32')
    
    #stack source and target indices
    idx = np.vstack((idx_source, idx_target))

    attributes = []
    for distances in distances_stack:
        # edge weights

        if not is_vector:
            attr = distances.flatten()
            attr = np.sqrt(attr)
        else:
            attr = distances.reshape((distances.shape[1]*distances.shape[0], distances.shape[2]))
            attr = np.expand_dims(attr, axis=0)

        attributes.append(attr)

    attributes = np.dstack(attributes)

    attributes = attributes[0]

    # into tensors
    x = torch.tensor(x, dtype = torch.float32)
    y = torch.tensor(y,dtype = torch.float32)
    idx = torch.tensor(idx, dtype = torch.long)
    attributes = torch.tensor(attributes, dtype = torch.float32)

    #build PyTorch geometric Data object
    data = Data(x = x, edge_index = idx, edge_attr = attributes, y = y)
    
    return data

########################################## DATASET FUNCTIONS ####################################   
#  
# split training data into train set and validation set
def split_data(seed, all_train_x, all_train_y, all_test_x, all_test_y):
    np.random.seed(seed)

    val_idx = np.random.choice(np.arange(len(all_train_x)),size = int(0.15*len(all_train_x)), replace = False)
    val_mask = np.zeros(len(all_train_x))
    val_mask[val_idx] = 1
    val_x = all_train_x[val_mask == 1]; val_y = all_train_y[val_mask == 1]
    train_x = all_train_x[val_mask == 0]; train_y = all_train_y[val_mask == 0]
    
    scaler = MinMaxScaler()
    scaler.fit(train_x[train_y == 0])
    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)
   
    if all_test_x is None:
        test_x = val_x
        test_y = val_y
    
    test_x = scaler.transform(all_test_x)
    test_y = all_test_y
	
    return train_x.astype('float32'), train_y.astype('float32'), val_x.astype('float32'), val_y.astype('float32'),  test_x.astype('float32'), test_y.astype('float32')


#load data
def load_dataset(dataset,seed):     
    np.random.seed(seed*100)
    
    if dataset == 'MI-V':
        df = pd.read_csv("data/MI/experiment_01.csv")
        for i in ['02','03','11','12','13','14','15','17','18']:
            data = pd.read_csv("data/MI/experiment_%s.csv" %i)
            # df = df.append(data, ignore_index = True)
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        normal_idx = np.ones(len(df))
        for i in ['06','08','09','10']:
            data = pd.read_csv("data/MI/experiment_%s.csv" %i)
            # df = df.append(data, ignore_index = True)
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
            normal_idx = np.append(normal_idx,np.zeros(len(data)))
        machining_process_one_hot = pd.get_dummies(df['Machining_Process'])
        df = pd.concat([df.drop(['Machining_Process'],axis=1),machining_process_one_hot],axis=1)
        data = df.to_numpy()
        idx = np.unique(data,axis=0, return_index = True)[1]
        data = data[idx]
        normal_idx = normal_idx[idx]
        normal_data = data[normal_idx == 1]
        anomaly_data = data[normal_idx == 0]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anomaly_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = np.zeros(len(train_x))
        test_x = np.concatenate((anomaly_data,normal_data[test_idx]))
        test_y  = np.concatenate((np.ones(len(anomaly_data)),np.zeros(len(test_idx))))

    elif dataset == 'MI-F':
        df = pd.read_csv("data/mi/experiment_01.csv")
        for i in ['02','03','06','08','09','10','11','12','13','14','15','17','18']:
            data = pd.read_csv("data/mi/experiment_%s.csv" %i)
            df = df.append(data, ignore_index = True)
        normal_idx = np.ones(len(df))
        for i in ['04', '05', '07', '16']: 
            data = pd.read_csv("data/mi/experiment_%s.csv" %i)
            df = df.append(data, ignore_index = True)
            normal_idx = np.append(normal_idx,np.zeros(len(data)))
        machining_process_one_hot = pd.get_dummies(df['Machining_Process'])
        df = pd.concat([df.drop(['Machining_Process'],axis=1),machining_process_one_hot],axis=1)
        data = df.to_numpy()
        idx = np.unique(data,axis=0, return_index = True)[1]
        data = data[idx]
        normal_idx = normal_idx[idx]
        normal_data = data[normal_idx == 1]
        anomaly_data = data[normal_idx == 0]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anomaly_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = np.zeros(len(train_x))
        test_x = np.concatenate((anomaly_data,normal_data[test_idx]))
        test_y  = np.concatenate((np.ones(len(anomaly_data)),np.zeros(len(test_idx))))  
        
    elif dataset in ['SAT', 'PEN', 'OPT', 'SPEECH', 'MNIST', 'ARR', 'OPTDIGITS', 'PENDIGITS','SHUTTLE', 'WBC','ANNT', 'THYR', 'MUSK', 'MAMO', 'ECOLI', 'VERT', 'WINE', 'BREAST', 'PIMA', 'GLASS']:
        if dataset == 'SHUTTLE':
            data = loadmat("./data/SHUTTLE/shuttle.mat")
        elif dataset == 'OPTDIGITS':
            data = loadmat("data/optdigits/optdigits.mat")
        elif dataset == 'PENDIGITS':
            data = loadmat('data/PENDIGITS/pendigits.mat')
        elif dataset == 'WBC':
            data = loadmat('./data/WBC/wbc.mat')
        elif dataset == 'ANNT':
            data = loadmat('./data/THYROID/annthyroid.mat')
        elif dataset == 'THYR':
            data = loadmat('./data/THYROID/thyroid.mat')
        elif dataset == 'MUSK':
            data = loadmat('./data/MUSK/musk.mat')
        elif dataset == 'ARR':
            data = loadmat('./data/ARR/arrhythmia.mat')
        elif dataset == 'MNIST':
            data = loadmat('./data/MNIST/mnist.mat')
        elif dataset == 'SPEECH':
            data = loadmat('./data/SPEECH/speech.mat')
        elif dataset == 'OPT':
            data = loadmat('./data/OPT/optdigits.mat')
        elif dataset == 'PEN':
            data = loadmat('./data/PEN/pendigits.mat')
        elif dataset == 'SAT':
            data = loadmat('./data/SAT/satellite.mat')
        elif dataset == 'MAMO':
            data = loadmat('./data/MAMO/mamo.mat')
        elif dataset == 'ECOLI':
            data = loadmat('./data/MAT/ecoli.mat')
        elif dataset == 'WINE':
            data = loadmat('./data/MAT/wine.mat')
        elif dataset == 'VERT':
            data = loadmat('./data/MAT/vertebral.mat')
        elif dataset == 'BREAST':
            data = loadmat('./data/MAT/breastw.mat')
        elif dataset == 'PIMA':
            data = loadmat('./data/MAT/pima.mat')
        elif dataset == 'GLASS':
            data = loadmat('./data/MAT/glass.mat')
        label = data['y'].astype('float32').squeeze()
        data = data['X'].astype('float32')
        normal_data= data[label == 0]
        normal_label = label[label==0]
        anom_data = data[label == 1]
        anom_label = label[label ==1]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anom_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = np.concatenate((normal_data[test_idx],anom_data))
        test_y = np.concatenate((normal_label[test_idx],anom_label))

        train_x = normal_data
        train_y = normal_label
        test_x = np.concatenate((normal_data, anom_data))
        test_y = np.concatenate((normal_label, anom_label))
        test_x = normal_data
        test_y = normal_label

        train_x, train_y = unison_shuffled_copies(train_x, train_y)
        test_x, test_y = unison_shuffled_copies(test_x, test_y)

    elif dataset in ['THYROID','HRSS']:
        if dataset == 'THYROID':
            data = pd.read_csv('./data/THYROID/annthyroid_21feat_normalised.csv').to_numpy()
        if dataset == 'HRSS':
            data = pd.read_csv('./data/HRSS/HRSS.csv').to_numpy()
        label = data[:,-1].astype('float32').squeeze()
        data = data[:,:-1].astype('float32')
        normal_data= data[label == 0]
        normal_label = label[label==0]
        anom_data = data[label == 1]
        anom_label = label[label ==1]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anom_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = np.concatenate((normal_data[test_idx],anom_data))
        test_y = np.concatenate((normal_label[test_idx],anom_label)) 
        
    elif dataset == 'SATELLITE':
        data = loadmat('data/SATELLITE/satellite.mat')
        label = data['y'].astype('float32').squeeze()
        data = data['X'].astype('float32')
        normal_data = data[label == 0]
        normal_label = label[label ==0]
        anom_data = data[label == 1]
        anom_label = label[label ==1]
        train_idx = np.random.choice(np.arange(0,len(normal_data)), 4000, replace = False)
        test_idx = np.setdiff1d(np.arange(0,len(normal_data)), train_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = normal_data[test_idx]
        test_y = normal_label[test_idx]
        test_idx = np.random.choice(np.arange(0,len(anom_data)), int(len(test_x)), replace = False)
        test_x = np.concatenate((test_x,anom_data[test_idx]))
        test_y = np.concatenate((test_y, anom_label[test_idx])) 
                
    train_x2, train_y2, val_x, val_y, test_x2, test_y2 = split_data(seed, all_train_x = train_x, all_train_y = train_y, all_test_x = test_x, all_test_y = test_y)
    # train_x, train_y, val_x, val_y, test_x, test_y = split_data(seed, all_train_x = train_x, all_train_y = train_y, all_test_x = test_x, all_test_y = test_y)

    # return train_x, train_y, test_x, test_y
    return normal_data, normal_label, anom_data, anom_label


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))