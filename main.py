import numpy as np
import time
import utils
import variables as var
from sklearn.metrics import roc_auc_score 
import LUNAR
import argparse
import matplotlib.pyplot as plt
import statistics
import pandas as pd

def main(args):
    datasets = ['WBC','ANNT', 'THYR', 'MUSK', 'MAMO', 'ECOLI', 'VERT', 'WINE', 'BREAST', 'PIMA', 'GLASS']
    datasets = ['WBC', 'GLASS', 'THYR']
    datasets = ['VERT']
    for dataset_arg in datasets:
        columns = ['method', 'seed', 'k', 'epoch', 'train_test', 'tp', 'tn', 'fp', 'fn']
        df = pd.DataFrame([], columns=columns)
        datas = []
        max_k = 20
        for seed in [0, 1, 2, 3, 4]:#range(1):
            # print("Running trial with random seed = %d" %seed)
            # load dataset (without negative samples)
            train_x, train_y, val_x, val_y, test_x, test_y = utils.load_dataset(dataset_arg, seed)
            datas.append(utils.negative_samples(train_x, train_y, val_x, val_y,test_x,test_y,max_k,args.samples,var.proportion,var.epsilon))

        d1 = []
        d2 = []
        d3 = []
        ks1 = []
        ks2 = []
        ks3 = []
        for mode in ['original and cof', 'original']:#, 'only cof']:
            for k in range(1, max_k+1):
                scores = []
                train_scores = []
                best_idxs = []
                for data in datas:
                    per_data_scores = []
                    per_data_train_scores = []
                    per_data_all_scores = []
                    x, y, neighbor_mask, train_mask, val_mask, test_mask, dist, dist2, idx = data
                    cut_dist, cut_dist2, cut_idx = utils.cut_data(dist, dist2, idx, k)
                    my_data = None
                    if mode == 'original and cof':
                        my_data = utils.build_graph(x, y, cut_idx, cut_dist, cut_dist2)
                    if mode == 'original':
                        my_data = utils.build_graph(x, y, cut_idx, cut_dist, cut_dist)
                    if mode == 'only cof':
                        my_data = utils.build_graph(x, y, cut_idx, cut_dist2, cut_dist2)
                    start = time.time()
                    test_scores, train_scores, all_scores, test_tps, test_tns, test_fps, test_fns, tps, tns, fps, fns = LUNAR.run(dataset_arg,seed,k,args.samples,args.train_new_model, my_data, train_mask, val_mask, test_mask)
                    end = time.time()
                    best_idx = 0
                    score = 0
                    scores.append([])
                    for test_out_idx in range(len(test_scores)):
                        new_score = 100*test_scores[test_out_idx]
                        new_train_score = 100*train_scores[test_out_idx]
                        new_all_score = 100*all_scores[test_out_idx]
                        per_data_scores.append(new_score)
                        per_data_train_scores.append(new_train_score)
                        per_data_all_scores.append(new_all_score)
                        scores[-1].append(new_score)
                        df.loc[len(df)] = [mode, seed, k, var.n_epochs*test_out_idx+1, 'train', tps[test_out_idx], tns[test_out_idx], fps[test_out_idx], fns[test_out_idx]]
                        df.loc[len(df)] = [mode, seed, k, var.n_epochs*test_out_idx+1, 'test', test_tps[test_out_idx], test_tns[test_out_idx], test_fps[test_out_idx], test_fns[test_out_idx]]

                        if new_score > score:
                            best_idx = test_out_idx
                            score = new_score
                    print('best idx: ' + str(best_idx))

                    best_idxs.append(best_idx)
                    ep_x = range(1, len(per_data_scores)+1)
                    plt.plot(ep_x, per_data_scores, 'r--', ep_x, per_data_train_scores, 'g--')
                    plt.show()

                mediane = statistics.mean(best_idxs)
                print('chosen idx: ' + str(int(mediane)))
                scr = 0
                for scrr in scores:
                    scr += scrr[int(mediane)]

                scr /= len(scores)

                if mode == 'original and cof':
                    ks1.append(k)
                    d1.append(scr)
                if mode == 'original':
                    ks2.append(k)
                    d2.append(scr)
                if mode == 'only cof':
                    ks3.append(k)
                    d3.append(scr)
                print('Mode: %s \t k: %d \t Score: %.4f' %(mode, k, scr))

        df.to_csv('results/' + str(dataset_arg) + '.csv')
        plt.plot(ks1, d1, 'r--', ks2, d2, 'b--', ks3, d3, 'g--')
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'HRSS')
    parser.add_argument("--samples", type = str, default = 'MIXED', help = 'Type of negative samples for training')
    parser.add_argument("--k", type = int, default = 100)
    parser.add_argument("--train_new_model", action="store_true", help = 'Train a new model vs. load existing model')
    args = parser.parse_args()

    main(args)
    