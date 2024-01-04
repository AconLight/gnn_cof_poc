# imports
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, confusion_matrix
from tqdm import tqdm

from src.gnn.gnn import GNN
from src.gnn.gnn_angle import GNNAngle
from src.gnn.gnn_layer import GNNLayer
from src.gnn.gnn_og_angle import GNNAngleOg
from src.gnn.gnn_og_tripple import GNNTrippleOg
from src.utils import variables as var


# Message passing scheme


def run(dataset,seed,k,datas, train_masks, test_masks, net):
    # print('train_new_model: ' + str(train_new_model))
    # loss function
    criterion = nn.MSELoss(reduction = 'none')    

    # path to save model parameters
    model_path = 'saved_models/%s/%d/net_%d.pth' %(dataset,k,seed)
    if not os.path.exists(os.path.dirname(model_path)):
       os.makedirs(os.path.dirname(model_path)) 
    
    # x, y, neighbor_mask, train_mask, val_mask, test_mask, dist, dist2, idx = utils.negative_samples(train_x, train_y, val_x, val_y, test_x, test_y, k, samples, var.proportion, var.epsilon, mode)
    # data = utils.build_graph(x, y, idx, dist, dist2)



    torch.manual_seed(seed)
    outs = []
    all_outs = []
    train_outs = []

    test_tps = []
    test_tns = []
    test_fps = []
    test_fns = []
    tps = []
    tns = []
    fps = []
    fns = []

    print("training GNN...")
    time.sleep(1)
    for epoches_ctr in tqdm(var.all_epoches):

        data = datas[epoches_ctr % len(datas)]
        data = data.to(var.device)
        train_mask = train_masks[epoches_ctr % len(train_masks)]
        test_mask = test_masks[epoches_ctr % len(test_masks)]

        optimizer = optim.Adam(net.parameters(), lr = var.lr, weight_decay = var.wd)

        # training
        for epoch in range(var.n_epochs):
            #print('epoch: ' + str(epoch))
            net.train()
            net.L1.is_train = True
            optimizer.zero_grad()
            out = net(data)
            # loss for training data only
            loss = criterion(out[train_mask == 1],data.y[train_mask == 1]).sum()
            loss.backward()
            optimizer.step()


        # testing
        with torch.no_grad():

            net.L1.is_train = False
            net.eval()
            out = net(data)
            loss = criterion(out,data.y)

        # outs.append(roc_auc_score(data.y[test_mask==1].cpu(), out[test_mask==1].cpu()))
        # train_outs.append(roc_auc_score(data.y[train_mask == 1].cpu(), out[train_mask == 1].cpu()))
        # all_outs.append(roc_auc_score(data.y[val_mask==1].cpu(),out[val_mask==1].cpu()))
        out21 = np.round(np.clip(out[test_mask==1].cpu().tolist(), 0, 1)).astype(bool)
        out22 = np.round(np.clip(out[train_mask==1].cpu().tolist(), 0, 1)).astype(bool)
        outs.append(precision_score(data.y[test_mask==1].cpu().tolist(), out21))
        train_outs.append(precision_score(data.y[train_mask == 1].cpu().tolist(), out22))

        test_tn, test_fp, test_fn, test_tp = confusion_matrix(data.y[test_mask==1].cpu().tolist(), out21).ravel()
        tn, fp, fn, tp = confusion_matrix(data.y[train_mask == 1].cpu().tolist(), out22).ravel()
        test_tps.append(test_tp)
        test_tns.append(test_tn)
        test_fps.append(test_fp)
        test_fns.append(test_fn)
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)

        # if len(outs) > 100 and np.average(train_outs[-100:-1]) - 0.15 < np.average(outs[-100:-1]) < np.average(train_outs[-100:-1]) - 0.03:
        #     break
    del net
    # return output for test points
    return outs, train_outs, all_outs, test_tps, test_tns, test_fps, test_fns, tps, tns, fps, fns
