
def plot2(dataset_arg, start_k, max_k, method_name_1, method_name_2):
    from pandas import read_csv
    from seaborn import lineplot
    from matplotlib import pyplot as plt
    # load the dataset
    # columns = ['method', 'seed', 'k', 'epoch', 'train_test', 'tp', 'tn', 'fp', 'fn']
    path = 'results/' + str(dataset_arg) + '.csv'
    dataset = read_csv(path, header=0)

    def draw(data_train, label1, data_test, label2, metric, title):
        plt.plot(data_train['epoch'], data_train[metric], label=label1)
        plt.plot(data_test['epoch'], data_test[metric], label=label2)
        plt.grid(True)
        plt.ylim(0, 1)
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.title(title)
        plt.savefig('results/plots/' + str(dataset_arg) + '_' + str(title) + '_' + metric + '.png')
        plt.close()

    def label_accuracy(row):
        return (row['tp'] + row['tn']) / (row['tp'] + row['tn'] + row['fp'] + row['fn'])

    data_train = dataset[dataset['train_test'] == 'train']
    data_test = dataset[dataset['train_test'] == 'test']

    data_train_og = data_train[data_train['method'] == method_name_1]
    data_test_og = data_test[data_test['method'] == method_name_1]
    data_train_new = data_train[data_train['method'] == method_name_2]
    data_test_new = data_test[data_test['method'] == method_name_2]
    og_best = 0
    new_best = 0
    og_best_k = 0
    new_best_k = 0
    og_best_stats = None
    new_best_stats = None

    for k in range(start_k, max_k + 1):  # [4,5,10]:
        data_train_og_k = data_train_og[data_train_og['k'] == k]
        data_test_og_k = data_test_og[data_test_og['k'] == k]
        data_train_new_k = data_train_new[data_train_new['k'] == k]
        data_test_new_k = data_test_new[data_test_new['k'] == k]
        # for seed in [0, 1, 2, 3, 4]:
        data_train_og_k_seeded = data_train_og_k.groupby(['epoch'], as_index=False)[['tp', 'tn', 'fp', 'fn']].sum()
        data_test_og_k_seeded = data_test_og_k.groupby(['epoch'], as_index=False)[['tp', 'tn', 'fp', 'fn']].sum()
        data_train_new_k_seeded = data_train_new_k.groupby(['epoch'], as_index=False)[['tp', 'tn', 'fp', 'fn']].sum()
        data_test_new_k_seeded = data_test_new_k.groupby(['epoch'], as_index=False)[['tp', 'tn', 'fp', 'fn']].sum()

        data_train_og_k_seeded['accuracy'] = data_train_og_k_seeded.apply(lambda row: label_accuracy(row), axis=1)
        data_test_og_k_seeded['accuracy'] = data_test_og_k_seeded.apply(lambda row: label_accuracy(row), axis=1)
        data_train_new_k_seeded['accuracy'] = data_train_new_k_seeded.apply(lambda row: label_accuracy(row), axis=1)
        data_test_new_k_seeded['accuracy'] = data_test_new_k_seeded.apply(lambda row: label_accuracy(row), axis=1)

        og_best_try = float(data_test_og_k_seeded.loc[data_test_og_k_seeded['accuracy'].idxmax()]['accuracy'])
        new_best_try = float(data_test_new_k_seeded.loc[data_test_new_k_seeded['accuracy'].idxmax()]['accuracy'])

        if og_best_try > og_best:
            og_best = og_best_try
            og_best_stats = data_test_og_k_seeded.loc[data_test_og_k_seeded['accuracy'].idxmax()]
            og_best_k = k

        if new_best_try > new_best:
            new_best = new_best_try
            new_best_stats = data_test_new_k_seeded.loc[data_test_new_k_seeded['accuracy'].idxmax()]
            new_best_k = k



        draw(data_train_og_k_seeded, 'train', data_test_og_k_seeded, 'test', 'accuracy', method_name_1 + ', k = ' + str(k))
        draw(data_train_new_k_seeded, 'train', data_test_new_k_seeded, 'test', 'accuracy', method_name_2 + ', k = ' + str(k))
        draw(data_test_new_k_seeded, method_name_2 + ' test', data_test_og_k_seeded, method_name_1 + ' test', 'accuracy',
             method_name_2 + ' vs ' + method_name_1 + ', k = ' + str(k))


    print(dataset_arg)
    print()
    print('k = ' + str(og_best_k))
    print(og_best_stats)
    print()
    print('k = ' + str(new_best_k))
    print(new_best_stats)








def plot3():
    import numpy as np
    import matplotlib.pyplot as plt
    # evenly sampled time at 200ms intervals
    k_list = list(range(1, 21)) + [31, 41, 50]
    d1 = [
        100.000,
        98.2180,
        98.8911,
        99.6351,
        99.5749,
        99.8335,
        99.8583,
        99.7839,
        99.9539,
        99.8902,
        99.7768,
        99.9575,
        99.9079,
        99.6918,
        99.9717,
        99.9752,
        99.9433,
        99.9646,
        99.9327,
        99.9433
    ]

    d2 = [
        100.000,
        99.9291,
        98.5404,
        98.0444,
        98.2145,
        97.9098,
        97.9736,
        97.9204,
        97.8885,
        97.8071,
        97.8637,
        97.7929,
        97.8850,
        97.8071,
        97.8177,
        97.6476,
        97.6724,
        97.7043,
        97.5662,
        97.4705,
    ]

    d3 = [
        100.000,
        100.000,
        98.4235,
        98.6857,
        97.2190,
        96.0286,
        95.7523,
        95.8727,
        91.5152,
        92.2982,
        89.9104,
        85.4926,
        84.6955,
        88.5960,
        88.5960,
        88.5960,
        88.5960,
        88.5960,
        88.5960,
        88.5960,
    ]
    # red dashes, blue squares and green triangles
    plt.plot(k_list, d1, 'r--', k_list, d2, 'b--', k_list, d3, 'g--')
    plt.show()
