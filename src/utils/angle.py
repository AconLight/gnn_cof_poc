import numpy as np
import torch

from src.utils import utils


def calc_pair_angles(inputs, k):
    message = []
    splited = np.split(inputs, int(inputs.shape[0] / k))
    for neighbors in splited:
        pairs = [(a, b) for idx, a in enumerate(neighbors) for b in neighbors[idx + 1:]]
        message_neighbors = []
        for pair in pairs:
            angle = utils.angle_between(pair[0], pair[1])
            message_neighbors.append(angle)
        message.append(message_neighbors)

    return torch.from_numpy(np.array(message)).to(torch.float32)

def calc_pair_angles_and_dist(inputs, k):
    message = []
    splited = np.split(inputs, int(inputs.shape[0] / k))
    for neighbors in splited:
        pairs = [(a, b) for idx, a in enumerate(neighbors) for b in neighbors[idx + 1:]]
        message_neighbors = []
        for pair in pairs:
            angle = utils.angle_between(pair[0], pair[1])
            message_neighbors.append(angle)
        for neighbor in neighbors:
            message_neighbors.append(np.linalg.norm(neighbor.numpy()))
        message.append(message_neighbors)

    return torch.from_numpy(np.array(message)).to(torch.float32)

