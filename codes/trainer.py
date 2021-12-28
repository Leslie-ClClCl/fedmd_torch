import logging

import numpy as np
import torch
from progressbar import *
from torch.utils import data

from codes.data_process import dataset_cifar
from codes.data_utils import get_dataset
from codes.utils import get_network, transform_train, transform_test

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >./tmp')
memory_gpu = [int(x.split()[2]) for x in open('./tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')


def trainer(net, data_loader, epoch, optimizer, criteria, warmup_scheduler, using_gpu=True, writer=None):
    start_time = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(data_loader):
        if using_gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(data_loader) + batch_index + 1
        last_layer = list(net.children())[-1]
        # update training loss for each iteration
        # writer.add_scalar('Train/loss', loss.item(), n_iter)
        if epoch <= 1:
            warmup_scheduler.step()
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        # writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    finish_time = time.time()
    logging.info('Training Epoch: {epoch} \tLoss: {:0.4f}\tLR: {:0.6f}\tTime Consumed:{:.2f}'.format(
        loss.item(),
        optimizer.param_groups[0]['lr'],
        finish_time - start_time,
        epoch=epoch))
    return loss.item


@torch.no_grad()
def evaluate_acc(net, data_loader, criteria=None, epoch=0, tb=False, using_gpu=True, writer=None):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in data_loader:

        if using_gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        if criteria is not None:
            loss = criteria(outputs, labels)
            test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    logging.info('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(data_loader.dataset),
        correct.float() / len(data_loader.dataset),
        finish - start
    ))
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(data_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(data_loader.dataset), epoch)

    return correct.float().item() / len(data_loader.dataset)



if __name__ == '__main__':
    net = get_network('resnet18', 10)
    net.load_state_dict(torch.load('/home/lichenglong/pycharm_project/fedmd_torch_new/results_imagenet_cifar/resnet18_200_10_t5/logit_matching_checkpoints/016.pth')['model_state_dict'])
    X_train_public, y_train_public, X_test_public, y_test_public = get_dataset('cifar10')
    data_set = dataset_cifar(X_train_public, y_train_public, transform_test)
    loader = data.DataLoader(data_set, batch_size=512, shuffle=True)
    acc = evaluate_acc(net, loader)
    print(acc)
    data_set = dataset_cifar(X_test_public, y_test_public, transform_test)
    loader = data.DataLoader(data_set, batch_size=512, shuffle=True)
    acc = evaluate_acc(net, loader)
    print(acc)