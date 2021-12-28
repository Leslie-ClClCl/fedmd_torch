import logging
import os.path

import cv2
import numpy
import numpy as np
import pandas as pd
from torchvision import datasets
from tqdm import tqdm


def tiny_imagenet_loader(root):
    trainset = datasets.ImageFolder(root=os.path.join(root, 'train'))
    testset = datasets.ImageFolder(root=os.path.join(root, 'val'))
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, pin_memory=True)
    train_image = [cv2.imread(fpath[0]) for fpath in trainset.samples]
    train_target = trainset.targets
    test_image = [cv2.imread(fpath[0]) for fpath in testset.samples]
    test_target = testset.targets
    return train_image, train_target, test_image, test_target


def get_dataset(dataset_name):
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST
    elif dataset_name == 'imagenet_tiny':
        train_data, train_target, test_data, test_target = \
            tiny_imagenet_loader('./data/tiny-imagenet-200/')
        train_data = numpy.array([cv2.resize(img, (32, 32)) for img in train_data])
        test_data = numpy.array([cv2.resize(img, (32, 32)) for img in test_data])
        return train_data, numpy.array(train_target), test_data, numpy.array(test_target)
    elif dataset_name == 'mura':
        data_frames = get_study_level_data_mura('XR_ELBOW')
        return data_frames['train'].iloc[:, 0:2], data_frames['train'].iloc[:, 2], \
               data_frames['valid'].iloc[:, 0:2], data_frames['valid'].iloc[:, 2],
    else:
        dataset = datasets.MNIST
    train_dataset = dataset(root='./data', train=True, download=True)
    test_dataset = dataset(root='./data', train=False, download=True)
    train_dataset.targets = np.squeeze(train_dataset.targets)
    test_dataset.targets = np.squeeze(test_dataset.targets)
    # add fake dimension for mnist dataset
    if dataset_name == 'mnist':
        train_dataset.data = train_dataset.data.numpy()
        test_dataset.data = test_dataset.data.numpy()
        train_dataset.targets = train_dataset.targets.numpy()
        test_dataset.targets = test_dataset.targets.numpy()
        train_dataset.data = numpy.array(
            [cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_GRAY2BGR) for img in train_dataset.data])
        test_dataset.data = numpy.array(
            [cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_GRAY2BGR) for img in test_dataset.data])
    return train_dataset.data, train_dataset.targets, test_dataset.data, test_dataset.targets


def generate_partial_data(X, y, class_in_use=None, N_total=0, verbose=False):
    if class_in_use is None:
        idx = np.ones_like(y, dtype=bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis=0)
    X_incomplete, y_incomplete = X[idx], y[idx]
    if N_total != 0:
        idx = numpy.random.choice(len(y), N_total)
        X_incomplete = X[idx]
        y_incomplete = y[idx]
    if verbose == True:
        logging.debug("X shape : {}".format(X_incomplete.shape))
        logging.debug("y shape : {}".format(y_incomplete.shape))
    return X_incomplete, y_incomplete


def generate_bal_private_data(X, y, N_parties=10, classes_in_use=range(11),
                              N_samples_per_class=20, data_overlap=False):
    priv_data = [None] * N_parties
    combined_idx = np.array([], dtype=np.int16)
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        idx = np.random.choice(idx, N_samples_per_class * N_parties,
                               replace=data_overlap)
        combined_idx = np.r_[combined_idx, idx]
        for i in range(N_parties):
            idx_tmp = idx[i * N_samples_per_class: (i + 1) * N_samples_per_class]
            if priv_data[i] is None:
                tmp = {}
                tmp["X"] = X[idx_tmp]
                tmp["y"] = y[idx_tmp]
                tmp["idx"] = idx_tmp
                priv_data[i] = tmp
            else:
                priv_data[i]['idx'] = np.r_[priv_data[i]["idx"], idx_tmp]
                priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]

    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return priv_data, total_priv_data


def get_study_level_data_mura(study_type):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes,
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    data_cat = ['train', 'valid']  # data categories
    for phase in data_cat:
        BASE_DIR = 'data/MURA-v1.1/MURA-v1.1/%s/%s/' % (phase, study_type)
        patients = list(os.walk(BASE_DIR))[0][1]  # list of patient folder names
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0
        for patient in tqdm(patients):  # for each patient folder
            for study in os.listdir(BASE_DIR + patient):  # for each study in that patient folder
                label = study_label[study.split('_')[1]]  # get label 0 or 1
                path = BASE_DIR + patient + '/' + study + '/'  # path to this study
                study_data[phase].loc[i] = [path, len(os.listdir(path)), label]  # add new row
                i += 1
    return study_data


if __name__ == '__main__':
    X = list(range(100))
    y = []
    for i in range(10):
        y.extend(list(range(10)))
    X, y = numpy.array(X), numpy.array(y)
    generate_partial_data(X, y, N_total=20)

