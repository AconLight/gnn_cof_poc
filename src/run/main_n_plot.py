import argparse

from src.plot.ploters import plot2
from src.run.main import main

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'HRSS')
    parser.add_argument("--samples", type = str, default = 'MIXED', help = 'Type of negative samples for training')
    parser.add_argument("--k", type = int, default = 100)
    parser.add_argument("--train_new_model", action="store_true", help = 'Train a new model vs. load existing model')
    args = parser.parse_args()

    main(args)

    dataset_arg = 'SAT'
    plot2(dataset_arg, 3, 3, 'original', 'original layer')