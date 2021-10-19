import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from data_process import generate_alignment_data, dataset_cifar, dataset_logits
from trainer import trainer, evaluate_acc
from trainer_logit import trainer_logits
from utils import WarmUpLR, transform_train, transform_test


def get_logits(net, dataloader):
    net.eval()
    ret = None
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()
            logits = net(x, logits=True, temperature=50)
            logits = np.array(logits.cpu())
            if ret is None:
                ret = logits
            else:
                ret = torch.cat((ret, logits), dim=0)
    return ret


class FedMD:
    def __init__(self, parties, ini_model, public_dataset, public_test_dataset,
                 private_data, total_private_data,
                 private_test_data, N_alignment,
                 N_rounds,
                 N_logits_matching_round, logits_matching_batchsize,
                 N_private_training_round, private_training_batchsize):
        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.public_test_data = public_test_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = N_alignment

        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize

        self.models = []
        self.init_result = []
        self.total_data = {}
        self.total_data["X"] = np.concatenate((self.public_dataset["X"], self.private_data[0]["X"]), axis=0)
        self.total_data["y"] = np.concatenate((self.public_dataset["y"], self.private_data[0]["y"]), axis=0)
        self.ini_model = ini_model
        public_test_set = dataset_cifar(public_test_dataset['X'], public_test_dataset['y'], transform=transform_test)
        self.public_test_loader = data.DataLoader(public_test_set, batch_size=128)
        self.private_test_loader = None
        self.private_train_loader = []

        for i in range(self.N_parties):
            train_set = dataset_cifar(self.private_data[i]['X'], self.private_data[i]['y'], transform=transform_train)
            train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True)
            self.private_train_loader.append(train_loader)
            test_set = dataset_cifar(self.private_test_data['X'], self.private_test_data['y'], transform=transform_test)
            test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False)
            self.private_test_loader = test_loader
            # checkpoint = torch.load('resnet18_cifar100_regular.pth')
            # parties[i].load_state_dict(checkpoint)

            criteria = nn.CrossEntropyLoss()
            optimizer = optim.SGD(parties[i].parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160],
                                                             gamma=0.2)  # learning rate decay
            iter_per_epoch = len(train_loader)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)
            best_acc = 0.0
            for epoch in range(1, 201):
                if epoch > 1:
                    train_scheduler.step()
                trainer(parties[i], train_loader, epoch, optimizer, criteria, warmup_scheduler)
                acc = evaluate_acc(parties[i], test_loader, criteria, epoch)

                # start to save best performance model after learning rate decay to 0.01
                if epoch > 120 and best_acc < acc:
                    weights_path = "resnet18_cifar100_best.pth"
                    print('saving weights file to {}'.format(weights_path))
                    torch.save(parties[i].state_dict(), weights_path)
                    best_acc = acc
                    continue

                if not epoch % 10:
                    weights_path = "resnet18_cifar100_regular.pth"
                    print('saving weights file to {}'.format(weights_path))
                    torch.save(parties[i].state_dict(), weights_path)

            self.models.append(parties[i])

    def collaborative_training(self):
        r = 0
        while True:
            # generate alignment data randomly
            alignment_set = generate_alignment_data(self.public_dataset['X'],
                                                    self.public_dataset['y'],
                                                    self.N_alignment, transform=transform_train)
            alignment_loader = data.DataLoader(alignment_set, batch_size=self.logits_matching_batchsize, shuffle=False)
            logging.info("round %d" % r)
            # get logits
            logits = 0
            for model in self.models:
                res = get_logits(model, alignment_loader).cpu()
                logits += res
            logits /= self.N_parties

            alignment_logit_set = dataset_logits(alignment_set.get_X(), logits, transform=transform_train)
            alignment_logit_loader = data.DataLoader(alignment_logit_set, shuffle=True, batch_size=128)
            r += 1
            if r > self.N_rounds:
                logging.info("##" * 16)
                # test parties' model performance
                for index, model in enumerate(self.models):
                    acc = evaluate_acc(self.private_test_loader, model)
                    logging.info("model {0} acc on private dataset: {1}".format(index, acc))
                    for cls in range(10, 22):
                        data_per_class_idx = (self.private_test_data["y"] == cls)
                        data_per_class_x = self.private_test_data["X"][data_per_class_idx]
                        data_per_class_y = self.private_test_data["y"][data_per_class_idx]
                        data_per_class_set = dataset_cifar(data_per_class_x, data_per_class_y)
                        data_per_class_loader = data.DataLoader(data_per_class_set, batch_size=128, shuffle=False)
                        acc = evaluate_acc(data_per_class_loader, model)
                        logging.info("model acc on {0} class : {1}".format(cls, acc))
                break

            # updates global model using logits
            # cause using only one party, do not update parties' model using logits
            # then train global model using public logits
            criteria = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-9)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.3)
            for epoch in range(300):
                trainer_logits(self.ini_model, train_loader=alignment_logit_loader, criterion=criteria,
                               optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            for cls in range(10, 22):
                data_per_class_idx = (self.private_test_data["y"] == cls)
                data_per_class_x = self.private_test_data["X"][data_per_class_idx]
                data_per_class_y = self.private_test_data["y"][data_per_class_idx]
                data_per_class_set = dataset_cifar(data_per_class_x, data_per_class_y)
                data_per_class_loader = data.DataLoader(data_per_class_set, batch_size=128, shuffle=False)
                acc_class = evaluate_acc(data_per_class_loader)
                logging.info("global model acc on {0} class : {1}".format(cls, acc_class))