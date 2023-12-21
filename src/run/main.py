import time

from src.plot.ploters import plot2
from src.gnn.gnn import GNN
from src.gnn.gnn_angle import GNNAngle
from src.gnn.gnn_og_angle import GNNAngleOg
from src.utils import variables as var, utils
from src.learning import LUNAR
import argparse
import statistics
import pandas as pd
import warnings


def main(args):
    warnings.filterwarnings("ignore")
    for dataset_arg in args.datasets.split("_"):
        columns = ['method', 'seed', 'k', 'epoch', 'train_test', 'tp', 'tn', 'fp', 'fn']
        results_df = pd.DataFrame([], columns=columns)
        datas = []
        for seed in [0]:#, 1, 2, 3, 4]:#range(1):
            train_x, train_y, val_x, val_y, test_x, test_y = utils.load_dataset(dataset_arg, seed)
            datas.append(utils.negative_samples(train_x, train_y, val_x, val_y,test_x,test_y,args.max_k,args.samples,var.proportion,var.epsilon))

        for mode in args.modes:
            for k in range(args.start_k, args.max_k+1):
                for data in datas:
                    x, y, neighbor_mask, train_mask, val_mask, test_mask, distances, distances_cof, distances_vectors, idx = data
                    cut_distances, cut_distances_cof, cut_distances_vectors, cut_idx = utils.cut_data(distances, distances_cof, distances_vectors, idx, k)
                    my_data = None
                    net = None
                    if mode == 'dist':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances])
                        net = GNN(k, k).to(var.device)
                    if mode == 'angle':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances_vectors], True)
                        net = GNNAngle(k, int(k*(k-1) / 2)).to(var.device)
                    if mode == 'dist_angle':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances_vectors], True)
                        net = GNNAngleOg(k, int(k*(k-1) / 2) + k).to(var.device)


                    test_scores, train_scores, all_scores, test_tps, test_tns, test_fps, test_fns, tps, tns, fps, fns = LUNAR.run(dataset_arg, seed, k, my_data, train_mask, val_mask, test_mask, net)
                    for test_out_idx in range(len(test_scores)):
                        results_df.loc[len(results_df)] = [mode, seed, k, var.n_epochs*test_out_idx+1, 'train', tps[test_out_idx], tns[test_out_idx], fps[test_out_idx], fns[test_out_idx]]
                        results_df.loc[len(results_df)] = [mode, seed, k, var.n_epochs*test_out_idx+1, 'test', test_tps[test_out_idx], test_tns[test_out_idx], test_fps[test_out_idx], test_fns[test_out_idx]]

        results_df.to_csv('results/' + str(dataset_arg) + '.csv')
        del datas, train_x, train_y, val_x, val_y, test_x, test_y
        del results_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=str, default = "MIXED")
    parser.add_argument("--start_k", type=int, default  = 3        )
    parser.add_argument("--max_k", type=int, default    = 5        )
    parser.add_argument('--datasets', type=str, default = "WBC_SAT") # split with _
    # ['HRSS', 'MI-V', 'MI-F', 'WBC','ANNT', 'THYR', 'MUSK', 'MAMO', 'VERT', 'WINE',
    #  'BREAST', 'PIMA', 'GLASS', 'MNIST', 'SPEECH', 'SAT', 'PEN', 'OPT', 'SHUTTLE']
    # ['WBC', 'MUSK', 'ARR', 'SPEECH', 'OPT', 'MNIST'] # high dimensions
    parser.add_argument('--modes', '--names-list', nargs='+', default=['dist', 'dist_angle'])
    # ['dist', 'angle', 'dist_angle']

    args = parser.parse_args()

    main(args)


    plot2(args.datasets.split('_'))
    