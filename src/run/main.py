import time

import numpy as np

from src.gnn.gnn_og_angle_var import GNNAngleOgVar
# import os
# import sys
# sys.path.append(os.getcwd())
# print(os.getcwd())
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
    var.proportion = args.proportion
    print(var.proportion)
    for dataset_arg in args.datasets.split("_"):
        print("----------------------------------------------------------")
        print(f"loading dataset: {dataset_arg}")
        columns = ['method', 'seed', 'k', 'epoch', 'train_test', 'tp', 'tn', 'fp', 'fn']
        results_df = pd.DataFrame([], columns=columns)
        datas = []
        sampling_methods = []
        datas_sizes = []
        for sampling_methods_arg in args.samples.split("_"):
            sampling_methods.append(sampling_methods_arg.split("-")[0])
            datas_sizes.append(int(sampling_methods_arg.split("-")[1]))


        for mode in args.modes:
            for k in range(args.start_k, args.max_k+1):
                datas = []
                for seed in range(len(datas_sizes)):
                    normal_x, normal_y, anom_x, anom_y = utils.load_dataset(dataset_arg, seed)
                    datas.append(utils.negative_samples(normal_x, normal_y, anom_x, anom_y, args.max_k,
                                                        sampling_methods[seed], var.proportion, var.epsilon))
                time.sleep(1)
                my_datas = []
                train_masks = []
                test_masks = []
                method_sampl = mode + "_" + args.samples.split("_")[seed]
                print(f"starting experiment (k: {k}, method: {method_sampl})")
                for data in datas:
                    x, y, neighbor_mask, train_mask, test_mask, distances, distances_cof, distances_vectors, idx = data
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
                    if mode == 'dist_angle_var':
                        my_data = utils.build_graph(x, y, cut_idx, [cut_distances_vectors], True)
                        net = GNNAngleOgVar(k, k+1).to(var.device)
                    my_datas.append(my_data)
                    train_masks.append(train_mask)
                    test_masks.append(test_mask)

                test_scores, train_scores, all_scores, test_tps, test_tns, test_fps, test_fns, tps, tns, fps, fns = LUNAR.run(dataset_arg, seed, k, my_datas, train_masks, test_masks, net)

                for test_out_idx in range(len(test_scores)):
                    results_df.loc[len(results_df)] = [method_sampl, 0, k, var.n_epochs*test_out_idx+1, 'train', tps[test_out_idx], tns[test_out_idx], fps[test_out_idx], fns[test_out_idx]]
                    results_df.loc[len(results_df)] = [method_sampl, 0, k, var.n_epochs*test_out_idx+1, 'test', test_tps[test_out_idx], test_tns[test_out_idx], test_fps[test_out_idx], test_fns[test_out_idx]]

        results_df.to_csv('results/' + str(dataset_arg) + '.csv')
        del datas
        del results_df


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=str, default = "MIXED-1_MIXED-5")
    # ['MIXED-1', 'MIXED-1_MIXED-5']
    parser.add_argument("--start_k", type=int, default  = 3        )
    parser.add_argument("--max_k", type=int, default    = 5        )
    parser.add_argument("--proportion", type=float, default    = 1.0        )
    parser.add_argument('--datasets', type=str, default = "WBC_SAT") # split with _
    # ['HRSS', 'MI-V', 'MI-F', 'WBC','ANNT', 'THYR', 'MUSK', 'MAMO', 'VERT', 'WINE',
    #  'BREAST', 'PIMA', 'GLASS', 'MNIST', 'SPEECH', 'SAT', 'PEN', 'OPT', 'SHUTTLE']
    # ['WBC', 'MUSK', 'ARR', 'SPEECH', 'OPT', 'MNIST'] # high dimensions
    parser.add_argument('--modes', '--names-list', nargs='+', default=['dist', 'dist_angle'])
    # ['dist', 'angle', 'dist_angle']

    args = parser.parse_args()

    main(args)

    datasets_plot = args.datasets.split('_')
    plot2(datasets_plot)

    print()
    print("finished in %s seconds ---" % int(time.time() - start_time))
    print()
    