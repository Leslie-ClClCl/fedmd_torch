import argparse
import copy
import logging
import os

import numpy as np
import pandas as pd
import torch.cuda

from FedMD import FedMD
from classifier_models.resnet import resnet18, resnet50
from data_utils import get_dataset, generate_partial_data, generate_bal_private_data
from utils import get_network

# select free GPU automatically


os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >./tmp')
memory_gpu = [int(x.split()[2]) for x in open('./tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')

CANDIDATE_MODELS = {"resnet_18": resnet18,
                    "resnet_50": resnet50,
                    }


def parseArgs():
    parser = argparse.ArgumentParser(description='FedMD')
    parser.add_argument('-models', type=str, action="append")
    parser.add_argument('-model_saved_dir', type=str, default='cifar_resnet18_15')
    parser.add_argument('-model_saved_names', type=str, action="append")
    parser.add_argument('-N_parties', type=int, default=1)
    parser.add_argument('-temperature', type=int, default=50)
    parser.add_argument('-N_samples_per_class', type=int, default=500)
    parser.add_argument('-N_alignment', type=int, default=40000)
    parser.add_argument('-private_classes', type=int)
    parser.add_argument('-public_classes', type=int)
    parser.add_argument('-N_rounds', type=int, default=1)
    parser.add_argument('-N_logits_matching_round', type=int, default=300)
    parser.add_argument('-N_private_training_round', type=int, default=4)
    parser.add_argument('-result_saved_path', type=str, default='results')
    parser.add_argument('-with_reverse', type=int, default=0)
    parser.add_argument('-train_private_model', type=int, default=1)
    parser.add_argument('-test', type=int, default=0)
    # parser.add_argument('-N_private', )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArgs()
    model_config = args.models
    model_saved_dir = args.model_saved_dir
    model_saved_names = args.model_saved_names

    public_classes = args.public_classes
    if public_classes == 10:
        public_classes = list(range(10))
    private_classes = args.private_classes
    model_saved_dir = model_saved_dir + '_' + str(private_classes)
    for idx, name in enumerate(model_saved_names):
        model_saved_names[idx] = name + '_' + str(private_classes) + 'cls'
    if private_classes == 6:
        private_classes = [3, 4, 13, 0, 5, 9]
    elif private_classes == 15:
        private_classes = [3, 13, 19, 58, 81, 8, 14, 77, 88, 92, 0, 5, 12, 36, 23]
    elif private_classes == 18:
        private_classes = [3, 4, 13, 19, 21, 30, 34, 41, 65, 0, 5, 9, 25, 39, 47, 51, 53, 54]
    elif private_classes == 30:
        private_classes = [3, 4, 13, 19, 21, 30, 31, 34, 41, 42, 65, 75, 89, 90, 91, 0, 5, 9, 17, 25, 28, 39, 47, 51,
                           53,
                           54, 56, 61, 62, 68]
    elif private_classes == 60:
        private_classes = [3, 4, 6, 7, 13, 15, 19, 21, 24, 26, 27, 30, 31, 32, 34, 38, 41, 42, 43, 65, 66, 74, 75, 80,
                           81, 88, 89, 90, 91, 97, 0, 5, 9, 10, 12, 17, 20, 23, 25, 28, 33, 37, 39, 40, 44, 45, 46, 47,
                           49, 51, 52, 53, 54, 56, 57, 59, 60, 61, 62, 68]
    else:
        private_classes = list(range(private_classes))

    n_classes = len(public_classes) + len(private_classes)

    N_parties = len(model_config)
    N_samples_per_class = 500

    N_rounds = args.N_rounds
    N_alignment = args.N_alignment
    N_private_training_round = args.N_private_training_round
    N_logits_matching_round = args.N_logits_matching_round
    private_training_batchsize = 5
    logits_matching_batchsize = 128
    temperature = args.temperature
    with_reverse = args.with_reverse
    result_saved_path = args.result_saved_path
    info = model_saved_names[0] + '-' + str(with_reverse)
    train_private_model = args.train_private_model
    test = args.test
    # logging.basicConfig(level=logging.INFO)
    if not os.path.exists(result_saved_path):
        os.mkdir(result_saved_path)
    logging.basicConfig(level=logging.DEBUG)

    # load data and create dataset
    X_train_CIFAR10, y_train_CIFAR10, X_test_CIFAR10, y_test_CIFAR10 = get_dataset('cifar10')
    X_train_CIFAR100, y_train_CIFAR100, X_test_CIFAR100, y_test_CIFAR100 = get_dataset('cifar100')
    # using only specified labeled data
    X_train_CIFAR100, y_train_CIFAR100 = generate_partial_data(X_train_CIFAR100, y_train_CIFAR100,
                                                               class_in_use=private_classes)
    X_test_CIFAR100, y_test_CIFAR100 = generate_partial_data(X_test_CIFAR100, y_test_CIFAR100,
                                                             class_in_use=private_classes)

    # relabel the targets of CIFAR-100
    y_tmp = copy.deepcopy(y_train_CIFAR100)
    y_test_tmp = copy.deepcopy(y_test_CIFAR100)
    for index, cls_ in enumerate(private_classes):
        y_train_CIFAR100[y_tmp == cls_] = index + len(public_classes)
        y_test_CIFAR100[y_test_tmp == cls_] = index + len(public_classes)
    del index, cls_
    logging.debug(pd.Series(y_train_CIFAR100).value_counts())
    mod_private_classes = np.arange(len(private_classes)) + len(public_classes)
    # create public dataset
    public_dataset = {"X": X_train_CIFAR10, "y": y_train_CIFAR10}
    public_test_dataset = {"X": X_train_CIFAR10, "y": y_test_CIFAR10}
    # create private dataset
    private_data, total_private_data \
        = generate_bal_private_data(X_train_CIFAR100, y_train_CIFAR100,
                                    N_parties=N_parties,
                                    classes_in_use=mod_private_classes,
                                    N_samples_per_class=N_samples_per_class,
                                    data_overlap=False)
    X_tmp, y_tmp = generate_partial_data(X=X_test_CIFAR100, y=y_test_CIFAR100,
                                         class_in_use=mod_private_classes)
    private_test_data = {"X": X_tmp, "y": y_tmp}
    logging.debug('data prepared!')
    # create classifier_models for each party
    parties = []
    if not os.path.exists(model_saved_dir):
        for i, model_name in enumerate(model_config):
            # tmp = CANDIDATE_MODELS[model_name](num_classes=n_classes, pretrained=True)
            tmp = get_network(model_name, num_classes=n_classes)
            tmp.cuda()
            logging.debug("model {0} : {1}".format(i, model_saved_names[i]))
            parties.append(tmp)
        # train_models(
        #     parties[0], X_train_CIFAR10, y_train_CIFAR10,X_test_CIFAR10, y_test_CIFAR10, epochs=5,
        #     save_dir=model_saved_dir, save_name=model_saved_names[0]+".pth")
    else:
        dpath = os.path.abspath(model_saved_dir)
        model_names = os.listdir(dpath)
        for idx, name in enumerate(model_names):
            model_name = model_config[idx]
            tmp = get_network(model_name, num_classes=n_classes)
            tmp.load_state_dict(torch.load(dpath + '/' + name)['state_dict'])
            parties.append(tmp)

    model_name = model_config[0]
    ini_model = get_network(model_name, num_classes=n_classes)
    ini_model.cuda()
    fedmd = FedMD(parties, ini_model=ini_model, public_dataset=public_dataset, public_test_dataset=public_test_dataset,
                  private_data=private_data, total_private_data=total_private_data, private_test_data=private_test_data,
                  N_rounds=N_rounds, N_alignment=N_alignment, N_logits_matching_round=N_logits_matching_round,
                  logits_matching_batchsize=logits_matching_batchsize,
                  N_private_training_round=N_private_training_round,
                  private_training_batchsize=private_training_batchsize)
    fedmd.collaborative_training()
