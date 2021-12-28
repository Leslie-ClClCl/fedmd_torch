import logging
import os
import time

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_target = sample
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def train(train_loader, model, criterion, optimizer, scheduler, epoch, temperature, result_saved_path=None):
    losses = AverageMeter()
    # switch to train mode
    model.train()

    iters = len(train_loader.dataset) // 128
    prefetcher = DataPrefetcher(train_loader)
    inputs, labels = prefetcher.next()
    iter_index = 1


    while inputs is not None:
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs, logits=True, temperature=temperature)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        inputs, labels = prefetcher.next()

        if iter_index % 100 == 0:
            logging.info(
                f"train:\tepoch {epoch:d},\titer [{iter_index:d}, \t{iters:d}],\tlr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f},\tloss_total: {loss.item():.6f}"
            )
        iter_index += 1
    scheduler.step(loss.item())
    return losses.avg


def trainer_logits(model, train_loader, criterion, optimizer, scheduler, epoch, temperature, save_dir='./', save_name=None):
    start_time = time.time()
    losses = train(train_loader, model, criterion, optimizer, scheduler, epoch, temperature)
    # remember best prec@1 and save checkpoint
    end_time = time.time()
    logging.info(
        f"val:\tepoch {epoch:d},\ttime consumed: {end_time - start_time}s, loss: {losses:.6f}"
    )
    torch.save(
        {
            'epoch': epoch,
            'loss': losses,
            'lr': optimizer.state_dict()['param_groups'][0]['lr'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(save_dir, save_name))
    return losses


def pred(model, data_loader):
    all_pred = None
    for idx, (img, label) in enumerate(data_loader):
        img, label = img.cuda(), label.cuda()
        pred = model(img)
        if all_pred == None:
            all_pred = pred
        else:
            all_pred = torch.cat([all_pred, pred], dim=0)
    return all_pred
