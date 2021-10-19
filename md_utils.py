import os.path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_process import dataset_cifar

# from torch.utils.tensorboard import SummaryWriter


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
MILESTONES = [10, 30, 60]


def train(net, epoch, dataloader, optimizer, criteria):
    start = time.time()
    net.train()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()
        del images, labels

    finish = time.time()
    print('Training Epoch: {epoch} \tLoss: {:0.4f}\tLR: {:0.6f}, time consumed{:.2f}s'.format(
        loss.item(),
        optimizer.param_groups[0]['lr'],
        finish - start,
        epoch=epoch))


def eval_training(net, epoch, dataloader, criteria):
    net.eval()
    test_loss = 0.0
    correct = 0.0

    for idx, (images, labels) in enumerate(dataloader):
        images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if idx == 1:
            print('break')
        loss = criteria(outputs, labels)
        # test_loss += loss
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        del images, labels, outputs, loss

    print('Epoch {}, Test acc: {:.4f}, Total loss{:.4f}'.format(
        epoch, correct.float() / len(dataloader.dataset), test_loss / len(dataloader.dataset)))
    return correct.float() / len(dataloader.dataset)


def train_models(net, X_train, y_train, X_test, y_test, epochs=20,
                 lr=0.1, log_dir=None, warms=1, save_dir=None, save_name=None
                 ):
    best_acc = 0.0
    training_dataset = dataset_cifar(X_train, y_train, training=True)
    training_loader = DataLoader(training_dataset, shuffle=True, num_workers=4, batch_size=64)

    test_dataset = dataset_cifar(X_test, y_test, training=False)
    test_loader = DataLoader(test_dataset, shuffle=True, num_workers=4, batch_size=64)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES,
                                                     gamma=0.1)  # learning rate decay
    iter_per_epoch = len(training_loader)

    for epoch in range(1, epochs + 1):
        if epoch > warms:
            train_scheduler.step()
        train(net, epoch, training_loader, optimizer, loss_function)
        acc = eval_training(net, epoch, test_loader, loss_function)
        if not os.path.exists(os.path.join(save_dir)):
            os.mkdir(os.path.join(save_dir))
        if epoch > MILESTONES[1] and best_acc < acc:
            torch.save({'state_dict': net.state_dict()}, os.path.join(save_dir, save_name))
            best_acc = acc
