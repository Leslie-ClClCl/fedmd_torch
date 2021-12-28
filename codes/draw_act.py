import copy

import numpy
import numpy as np
import torch
import torch.nn as nn
from data_utils import get_dataset, generate_partial_data
from utils import get_network, transform_train
from data_process import dataset_cifar
import torch.utils.data as data
from models.resnet import BasicBlock
from draw_hist import draw_hist, draw_activated
import pandas as pd

def get_act_out(net, input):
    output = []
    blocks = [net.conv1, net.conv2_x, net.conv3_x, net.conv4_x, net.conv5_x]
    x = input
    for block in blocks:
        for func in list(block):
            if isinstance(func, BasicBlock):
                x1 = copy.deepcopy(x.detach())
                for sub_func in list(func.residual_function): # residual functions
                    x = sub_func(x)
                    if isinstance(sub_func, nn.ReLU):
                        output.append(x)
                for sub_func in list(func.shortcut): # residual functions
                    x1 = sub_func(x1)
                    if isinstance(sub_func, nn.ReLU):
                        output.append(x1)
                x = nn.ReLU(inplace=True)(x + x1)
                output.append(x)
                del x1
            else:
                x = func(x)
                if isinstance(func, nn.ReLU):
                    output.append(x)

    return output


def get_activated_num(activate):
    num = torch.count_nonzero(activate)
    return num

def save_csv(data_list, filename, sheet_names):
    for idx, data in enumerate(data_list):
        data = [item.flatten() for item in data]
        df = pd.DataFrame(data)
        df_T = pd.DataFrame(df.values.T)
        df_T.to_csv(sheet_names[idx]+filename)
    print('Data has saved to '+filename)


def draw_activate_num(ref_model_num, logit_matching_num, sample_num):
    shape = [65536, 65536, 65536, 65536, 65536, 32768, 32768, 32768, 32768, 16384, 16384, 16384, 16384, 8192, 8192,
             8192, 8192]
    shape = np.array(shape)
    logit_matching_num, ref_model_num = logit_matching_num / 50000, ref_model_num / 50000
    logit_matching_num, ref_model_num = logit_matching_num / shape, ref_model_num / shape
    draw_hist([ref_model_num, logit_matching_num], list(range(17)), ['ref', 'logit_matching'])

if __name__ == '__main__':
    model_name = 'resnet18'
    N_classes = 10
    state_dict_path_ref = '/home/lichenglong/pycharm_project/fedmd_torch_new/results_imagenet_cifar/checkpoints/resnet18_200_10/resnet18_10cls.pth'
    state_dict_path_logit_matching = '/home/lichenglong/pycharm_project/fedmd_torch_new/results_THU/resnet18_10_10_t5/logit_matching_checkpoints/THU.pth'
    net_logit = get_network(model_name, N_classes)
    net_logit.load_state_dict(torch.load(state_dict_path_logit_matching)['model_state_dict'])

    net_ref = get_network(model_name, N_classes)
    net_ref.load_state_dict(torch.load(state_dict_path_ref))

    X_train, y_train, X_test, y_test = get_dataset('cifar10')
    X_total = np.append(X_train, X_test, axis=0)
    y_total = np.append(y_train, y_test, axis=0)
    # CIFAR 10 Train
    # X_train, y_train = generate_partial_data(X_train, y_train, N_total=10000, class_in_use=[5, 7])
    data_set = dataset_cifar(X_train, y_train, transform_train)
    loader = data.DataLoader(data_set, batch_size=512, shuffle=True)
    N_activate_CIFAR_Train_logit = numpy.array([0] * 17)
    N_activate_CIFAR_Train_ref = numpy.array([0] * 17)
    for idx, (x, y) in enumerate(loader):
        input = x.cuda()
        output_logit = get_act_out(net_logit, input)
        output_ref = get_act_out(net_ref, input)
        # if idx == 0:
        #     N_activate_CIFAR_Train = [np.array(torch.count_nonzero(item.cpu(), dim=0)) for item in output]
        # else:
        #     N_activate_CIFAR_Train = [np.array(torch.count_nonzero(item.cpu(), dim=0)) +
        #                               N_activate_CIFAR_Train[idx] for idx, item in enumerate(output)]
        output_logit = numpy.array([get_activated_num(act).item() for act in output_logit])
        output_ref = numpy.array([get_activated_num(act).item() for act in output_ref])
        N_activate_CIFAR_Train_logit = N_activate_CIFAR_Train_logit + output_logit
        N_activate_CIFAR_Train_ref = N_activate_CIFAR_Train_ref + output_ref
        print(idx)
    draw_activate_num(np.array(N_activate_CIFAR_Train_ref), np.array(N_activate_CIFAR_Train_logit), 50000)


    # N_activate_CIFAR_Train = [item/10000 for item in N_activate_CIFAR_Train]
    # draw_activated([N_activate_CIFAR_Train[0]], [list(range(65536))])
    # save_csv([N_activate_CIFAR_Train], '_origin_model.csv', ['BAD_CLASSES_5_7'])
    # exit()
    # CIFAR 10 Test
    data_set = dataset_cifar(X_test, y_test, transform_train)
    loader = data.DataLoader(data_set, batch_size=512, shuffle=True)
    N_activate_CIFAR_Test = numpy.array([0] * 17)
    for idx, (x, y) in enumerate(loader):
        input = x.cuda()
        output = get_act_out(net, input)
        if idx == 0:
            N_activate_CIFAR_Test = [np.array(torch.count_nonzero(item.cpu(), dim=0)) for item in output]
        else:
            N_activate_CIFAR_Test = [np.array(torch.count_nonzero(item.cpu(), dim=0)) +
                                      N_activate_CIFAR_Test[idx] for idx, item in enumerate(output)]
        # output = numpy.array([get_activated_num(act).item() for act in output])
        # N_activate_CIFAR_Test = N_activate_CIFAR_Test + output
        print(idx)
    # N_activate_CIFAR_Test = [item/10000 for item in N_activate_CIFAR_Test]

    # ImageNet Mini
    X_train_img, y_train_img, X_test_img, y_test_img = get_dataset('imagenet_tiny')
    data_set = dataset_cifar(X_train_img, y_train_img, transform_train)
    loader = data.DataLoader(data_set, batch_size=512, shuffle=True)
    N_activate_ImageNet_train = numpy.array([0] * 17)
    for idx, (x, y) in enumerate(loader):
        input = x.cuda()
        output = get_act_out(net, input)
        if idx == 0:
            N_activate_ImageNet_train = [np.array(torch.count_nonzero(item.cpu(), dim=0)) for item in output]
        else:
            N_activate_ImageNet_train = [np.array(torch.count_nonzero(item.cpu(), dim=0)) +
                                      N_activate_ImageNet_train[idx] for idx, item in enumerate(output)]
        # output = numpy.array([get_activated_num(act).item() for act in output])
        # N_activate_ImageNet_train = N_activate_ImageNet_train + output
        print(idx)
    N_activate_ImageNet_train = [item/100000 for item in N_activate_ImageNet_train]

    # save_csv([N_activate_CIFAR_Train, N_activate_CIFAR_Test, N_activate_ImageNet_train],
    #          '_origin_model.csv',
    #          ['CIFAR_Train', 'CIFAR_Test', 'ImageNet_Train'])
    draw_activated([N_activate_CIFAR_Train[0], N_activate_CIFAR_Test[0], N_activate_ImageNet_train[0]],
                   list(range(65536)))
    # draw_hist([N_activate_CIFAR_Train, N_activate_CIFAR_Test, N_activate_ImageNet_train], list(range(17)),
    #           ['CIFAR-10 Train', 'CIFAR-10 Test', 'ImageNet Train'])

    # imagenet cifar
    # shape = [65536, 65536, 65536, 65536, 65536, 32768, 32768, 32768, 32768, 16384, 16384, 16384, 16384, 8192, 8192,
    #          8192, 8192]
    # shape = np.array(shape)
    # logit_matching = np.array([2333271146, 2063579212, 2291799179, 1424741649, 2470374869, 715737222, 806191972, 604232164,
    #                   982111263, 296517543, 352285796, 239643810, 445289033, 112971138, 92525492, 125451726, 131725703])
    # ref_model = np.array([764963595, 1648972441, 1319596048, 1218959346, 2610232444, 614148196, 616190866, 316536321, 739683631,
    #              270591172, 179637779, 142064707, 191795803, 105224137, 156852716, 164559285, 256733758])
    # logit_matching, ref_model = logit_matching / 60000, ref_model / 60000
    # logit_matching, ref_model = logit_matching / shape, ref_model / shape
    # draw_hist([ref_model, logit_matching], list(range(17)), ['ref', 'logit_matching'])

    # cifar10


