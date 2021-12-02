import argparse
import logging
import os
import time

import numpy as np
import torch.cuda

from FedMD import FedMD
from data_utils import get_dataset, generate_partial_data, generate_bal_private_data
from draw_hist import draw_hist, fine_label_names
from utils import get_network

# select free GPU automatically


os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >./tmp')
memory_gpu = [int(x.split()[2]) for x in open('./tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.system('rm tmp')


def parseArgs():
    parser = argparse.ArgumentParser(description='FedMD')
    parser.add_argument('-models', type=str, action="append")
    parser.add_argument('-N_parties', type=int, default=1)
    parser.add_argument('-temperature', type=int, default=10)
    parser.add_argument('-N_samples_per_class', type=int, default=500)
    parser.add_argument('-N_alignment', type=int, default=40000)
    parser.add_argument('-private_classes', type=int)
    parser.add_argument('-public_classes', type=int, default=10)
    parser.add_argument('-N_rounds', type=int, default=1)
    parser.add_argument('-N_logits_matching_round', type=int, default=300)
    parser.add_argument('-N_private_training_round', type=int, default=200)
    parser.add_argument('-result_saved_path', type=str, default='results/')
    parser.add_argument('-with_reverse', type=int, default=0)
    parser.add_argument('-train_private_model', type=int, default=1)
    # parser.add_argument('-N_private', )

    args = parser.parse_args()
    return args


def parse_data(public_dataset_name, private_dataset_name, public_classes_in_use, private_classes_in_use,
               N_parties, N_samples_per_class):
    # load data and create dataset
    X_train_public, y_train_public, X_test_public, y_test_public = get_dataset(public_dataset_name)
    X_train_private, y_train_private, X_test_private, y_test_private = get_dataset(private_dataset_name)

    # filter specific class
    X_train_public, y_train_public = generate_partial_data(X_train_public, y_train_public,
                                                           class_in_use=public_classes_in_use)
    X_test_public, y_test_public = generate_partial_data(X_test_public, y_test_public,
                                                         class_in_use=public_classes_in_use)

    X_train_private, y_train_private = generate_partial_data(X_train_private, y_train_private,
                                                             class_in_use=private_classes_in_use)
    X_test_private, y_test_private = generate_partial_data(X_test_private, y_test_private,
                                                           class_in_use=private_classes_in_use)

    # relabel the targets of private data
    y_train_private = y_train_private + len(public_classes_in_use)
    y_test_private = y_test_private + len(public_classes_in_use)
    private_classes_in_use = [cls + len(public_classes_in_use) for cls in private_classes_in_use]

    # create public & private dataset
    public_dataset = {"X": X_train_public, "y": y_train_public}
    public_test_dataset = {"X": X_test_public, "y": y_test_public}

    private_data, total_private_data \
        = generate_bal_private_data(X_train_private, y_train_private,
                                    N_parties=N_parties,
                                    classes_in_use=private_classes_in_use,
                                    N_samples_per_class=N_samples_per_class,
                                    data_overlap=False)
    private_test_data = {"X": X_test_private, "y": y_test_private}
    return public_dataset, public_test_dataset, private_data, private_test_data, total_private_data


def fedmd_train():
    args = parseArgs()
    model_config = args.models

    public_classes = args.public_classes
    private_classes = args.private_classes
    result_saved_path = os.path.join(args.result_saved_path, model_config[0] + '_' + str(public_classes) + '_'
                                     + str(private_classes) + '_t' + str(args.temperature))
    if not os.path.exists(result_saved_path):
        os.mkdir(result_saved_path)
    model_saved_dir = os.path.join(args.result_saved_path, 'checkpoints', model_config[0] + '_' + str(public_classes)
                                   + '_' + str(private_classes))
    if not os.path.exists(model_saved_dir):
        os.mkdir(model_saved_dir)

    model_saved_names = []
    for idx, model_name in enumerate(model_config):
        model_saved_names.append(model_name + '_' + str(private_classes) + 'cls')

    public_classes = list(range(public_classes))
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
    N_samples_per_class = 5000

    N_rounds = args.N_rounds
    N_alignment = args.N_alignment
    N_private_training_round = args.N_private_training_round
    N_logits_matching_round = args.N_logits_matching_round
    private_training_batchsize = 5
    logits_matching_batchsize = 128
    temperature = args.temperature
    with_reverse = args.with_reverse

    info = model_saved_names[0] + '-' + str(with_reverse)
    train_private_model = args.train_private_model

    logging.basicConfig(level=logging.DEBUG)

    public_dataset, public_test_dataset, private_data, private_test_data, total_private_data \
        = parse_data('imagenet_tiny', 'cifar10', public_classes, private_classes, N_parties, N_samples_per_class)

    # public_dataset, public_test_dataset, private_data, private_test_data, total_private_data \
    #     = parse_data('mura', 'cifar10', public_classes, private_classes, N_parties, N_samples_per_class)

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
        if len(model_names) == 0:
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
            for idx, name in enumerate(model_names):
                model_name = model_config[idx]
                tmp = get_network(model_name, num_classes=n_classes)
                tmp.load_state_dict(torch.load(dpath + '/' + name))
                parties.append(tmp)

    model_name = model_config[0]
    ini_model = get_network(model_name, num_classes=n_classes)
    ini_model.cuda()
    fedmd = FedMD(parties, ini_model=ini_model, public_dataset=public_dataset, public_test_dataset=public_test_dataset,
                  private_data=private_data, total_private_data=total_private_data, private_test_data=private_test_data,
                  N_rounds=N_rounds, N_alignment=N_alignment, N_logits_matching_round=N_logits_matching_round,
                  logits_matching_batchsize=logits_matching_batchsize, model_saved_name=model_saved_names,
                  result_saved_dir=result_saved_path, N_public_classes=len(public_classes),
                  N_private_training_round=N_private_training_round, model_saved_dir=model_saved_dir,
                  private_training_batchsize=private_training_batchsize, temperature=temperature,
                  N_private_classes=len(private_classes), train_private_model=train_private_model)
    acc_ref, acc_ini = fedmd.collaborative_training()
    logging.info(acc_ref)
    logging.info(acc_ini)
    label_names_need = []
    for idx in private_classes:
        label_names_need.append(fine_label_names[idx] + ' ' + str(idx))
    draw_hist([acc_ref, acc_ini], label_names_need, model_saved_names[0],
              os.path.join(result_saved_path, model_saved_names[0] + '_' + time.strftime("%a %b %d %H:%M:%S %Y %Z",
                                                                                         time.localtime())) + '.png')


if __name__ == '__main__':
    fedmd_train()
    pass
