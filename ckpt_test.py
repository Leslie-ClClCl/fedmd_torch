import copy

import numpy as np
import torch
import torch.utils.data as data

from data_process import dataset_cifar
from data_utils import get_dataset
from utils import get_network, transform_test

tmp = get_network('resnet18', num_classes=110)
tmp.load_state_dict(
    torch.load('results_imagenet_cifar/resnet18_10_t50/logit_matching_checkpoints/latest.pth')['model_state_dict'])
tmp.cpu()


def get_test_data(dataset_name, private_classes):
    X_train, y_train, X_test, y_test = get_dataset(dataset_name)
    y_test = relabel_data(y_test, private_classes)
    test_set = dataset_cifar(X_test, y_test, transform=transform_test)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=True)
    return test_loader, X_test, y_test


def relabel_data(label, private_classes):
    label_tmp = copy.deepcopy(label)
    for index, cls_ in enumerate(private_classes):
        label[label_tmp == cls_] = index + len(private_classes)
    return label


private_classes = list(range(100))
test_loader, X_test, y_test = get_test_data('cifar10', private_classes=private_classes)
pred = None
for x, y in test_loader:
    pred_batch = tmp(x)
    pred_batch = pred_batch.detach().tolist()
    pred_batch = [one_hot.index(max(one_hot)) for one_hot in pred_batch]
    if pred is None:
        pred = pred_batch
    else:
        pred.extend(pred_batch)

pred = np.array(pred)
# C
for cls in range(100, 110):
    data_per_class_idx = [pred == cls]
    data_per_class_x = X_test[data_per_class_idx]
    data_per_class_y = y_test[data_per_class_idx]
    acc = np.mean(data_per_class_y == cls)
    print(acc)
# P
for cls in range(100, 110):
    data_per_class_idx = [y_test == cls]
    data_per_class_x = X_test[data_per_class_idx]
    data_per_class_y = y_test[data_per_class_idx]
    acc = np.mean(pred == cls)
    print(acc)
pass
