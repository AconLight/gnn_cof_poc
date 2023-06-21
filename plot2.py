from pandas import read_csv
from seaborn import lineplot
from matplotlib import pyplot as plt
# load the dataset
columns = ['method', 'seed', 'k', 'epoch', 'train_test', 'tp', 'tn', 'fp', 'fn']
dataset_arg = 'THYR'
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

def label_accuracy (row):
    return (row['tp'] + row['tn']) / (row['tp'] + row['tn'] + row['fp'] + row['fn'])

data_train = dataset[dataset['train_test'] == 'train']
data_test = dataset[dataset['train_test'] == 'test']

data_train_og = data_train[data_train['method'] == 'original']
data_test_og = data_test[data_test['method'] == 'original']
data_train_new = data_train[data_train['method'] == 'original and cof']
data_test_new = data_test[data_test['method'] == 'original and cof']

max_k = 20
for k in [4,5,10]:#range(1, max_k+1):
    data_train_og_k = data_train_og[data_train_og['k'] == k]
    data_test_og_k = data_test_og[data_test_og['k'] == k]
    data_train_new_k = data_train_new[data_train_new['k'] == k]
    data_test_new_k = data_test_new[data_test_new['k'] == k]
    # for seed in [0, 1, 2, 3, 4]:
    data_train_og_k_seeded = data_train_og_k.groupby(['epoch'], as_index=False)['tp', 'tn', 'fp', 'fn'].sum()
    data_test_og_k_seeded = data_test_og_k.groupby(['epoch'], as_index=False)['tp', 'tn', 'fp', 'fn'].sum()
    data_train_new_k_seeded = data_train_new_k.groupby(['epoch'], as_index=False)['tp', 'tn', 'fp', 'fn'].sum()
    data_test_new_k_seeded = data_test_new_k.groupby(['epoch'], as_index=False)['tp', 'tn', 'fp', 'fn'].sum()

    data_train_og_k_seeded['accuracy'] = data_train_og_k_seeded.apply(lambda row: label_accuracy(row), axis=1)
    data_test_og_k_seeded['accuracy'] = data_test_og_k_seeded.apply(lambda row: label_accuracy(row), axis=1)
    data_train_new_k_seeded['accuracy'] = data_train_new_k_seeded.apply(lambda row: label_accuracy(row), axis=1)
    data_test_new_k_seeded['accuracy'] = data_test_new_k_seeded.apply(lambda row: label_accuracy(row), axis=1)

    print('k' + str(k))
    print(data_test_og_k_seeded.loc[data_test_og_k_seeded['accuracy'].idxmax()])
    print(data_test_new_k_seeded.loc[data_test_new_k_seeded['accuracy'].idxmax()])

    if k == 12:
        asd = 2
    #draw(data_train_og_k_seeded, 'train', data_test_og_k_seeded, 'test', 'accuracy', 'original, k = ' + str(k))
    #draw(data_train_new_k_seeded, 'train', data_test_new_k_seeded, 'test', 'accuracy', 'new method, k = ' + str(k))
    #draw(data_test_new_k_seeded, 'new method test', data_test_og_k_seeded, 'original method test', 'accuracy', 'new method vs original method, k = ' + str(k))
    # create line plot







