import time
from src.utils import variables as var, utils
from src.learning import LUNAR
import argparse
import statistics
import pandas as pd

def main(args):
    datasets = ['WBC','ANNT', 'THYR', 'MUSK', 'MAMO', 'VERT', 'WINE', 'BREAST', 'PIMA', 'GLASS', 'MNIST', 'SPEECH', 'SAT', 'PEN', 'OPT', 'SHUTTLE']
    # datasets = ['THYR', 'WBC']
    # datasets = ['MUSK', 'GLASS', 'THYR', 'WBC']
    # datasets = ['ANNT', 'MAMO', 'VERT']
    # datasets = ['SHUTTLE', 'SAT', 'PEN', 'OPT', 'THYR'] # those are in LUNAR paper
    datasets = ['WBC']
    # datasets = ['WBC', 'MUSK', 'ARR', 'SPEECH', 'OPT', 'MNIST'] # high dimensions
    for dataset_arg in datasets:
        columns = ['method', 'seed', 'k', 'epoch', 'train_test', 'tp', 'tn', 'fp', 'fn']
        results_df = pd.DataFrame([], columns=columns)
        datas = []
        start_k = 3
        max_k = 7
        gnn_name = 'normal'
        for seed in [0]:#, 1, 2, 3, 4]:#range(1):
            # print("Running trial with random seed = %d" %seed)
            # load dataset (without negative samples)
            train_x, train_y, val_x, val_y, test_x, test_y = utils.load_dataset(dataset_arg, seed)
            datas.append(utils.negative_samples(train_x, train_y, val_x, val_y,test_x,test_y,max_k,args.samples,var.proportion,var.epsilon))
            print("loaded dataset")

        d1 = []
        d2 = []
        d3 = []
        ks1 = []
        ks2 = []
        ks3 = []
        # 36790
        # 7358
        # for mode in ['original and cof', 'only vectors']:#, 'only cof']:
        # for mode in ['only angle fit']:#, 'only cof']:
        for mode in ['original', 'only angles']:#, 'only cof']:
        # for mode in ['only angles']:#, 'only cof']:
        # for mode in ['only vectors']:#, 'only cof']:
            for k in range(start_k, max_k+1):
                scores = []
                train_scores = []
                best_idxs = []
                for data in datas:
                    per_data_scores = []
                    per_data_train_scores = []
                    per_data_all_scores = []
                    x, y, neighbor_mask, train_mask, val_mask, test_mask, distances, distances_cof, distances_vectors, idx = data
                    cut_distances, cut_distances_cof, cut_distances_vectors, cut_idx = utils.cut_data(distances, distances_cof, distances_vectors, idx, k)
                    my_data = None
                    gnn_name = 'normal'
                    if mode == 'original and cof':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances, cut_distances_cof])
                        input_size = k*2
                    if mode == 'original':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances])
                        input_size = k
                    if mode == 'only cof':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances_cof])
                    if mode == 'only vectors':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances_vectors], True)
                        input_size = k*x.shape[1]
                    if mode == 'only angles':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances_vectors], True)
                        input_size = int(k*(k-1) / 2)
                        gnn_name = 'angle'
                    if mode == 'original angle':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances_vectors], True)
                        input_size = int(k*(k-1) / 2) + k
                        gnn_name = 'original angle'
                    if mode == 'only angle fit':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances_vectors], True)
                        input_size = int(k / 2)
                        gnn_name = 'angle fit'
                    if mode == 'original and vectors':
                        pass
                        # my_data = utils.build_graph(x, y, cut_idx, [cut_distances, cut_distances_vectors])
                        # input_size = k*2

                    start = time.time()
                    test_scores, train_scores, all_scores, test_tps, test_tns, test_fps, test_fns, tps, tns, fps, fns = LUNAR.run(dataset_arg, seed, k, args.samples, args.train_new_model, my_data, train_mask, val_mask, test_mask, input_size, gnn_name)
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
                        results_df.loc[len(results_df)] = [mode, seed, k, var.n_epochs*test_out_idx+1, 'train', tps[test_out_idx], tns[test_out_idx], fps[test_out_idx], fns[test_out_idx]]
                        results_df.loc[len(results_df)] = [mode, seed, k, var.n_epochs*test_out_idx+1, 'test', test_tps[test_out_idx], test_tns[test_out_idx], test_fps[test_out_idx], test_fns[test_out_idx]]

                        if new_score > score:
                            best_idx = test_out_idx
                            score = new_score
                    print('best idx: ' + str(best_idx))

                    best_idxs.append(best_idx)
                    ep_x = range(1, len(per_data_scores)+1)
                    # plt.plot(ep_x, per_data_scores, 'r--', ep_x, per_data_train_scores, 'g--')
                    # plt.show()

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

        results_df.to_csv('results/' + str(dataset_arg) + '.csv')
        # plt.plot(ks1, d1, 'r--', ks2, d2, 'b--', ks3, d3, 'g--')
        # plt.show()
        del d1, d2, d3, ks1, ks2, ks3
        del datas, train_x, train_y, val_x, val_y, test_x, test_y
        del results_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'HRSS')
    parser.add_argument("--samples", type = str, default = 'MIXED', help = 'Type of negative samples for training')
    parser.add_argument("--k", type = int, default = 100)
    parser.add_argument("--train_new_model", action="store_true", help = 'Train a new model vs. load existing model')
    args = parser.parse_args()

    main(args)
    