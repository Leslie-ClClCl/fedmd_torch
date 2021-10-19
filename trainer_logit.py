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


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    losses = AverageMeter()
    # switch to train mode
    model.train()

    iters = len(train_loader.dataset) // 128
    prefetcher = DataPrefetcher(train_loader)
    inputs, labels = prefetcher.next()
    iter_index = 1

    while inputs is not None:
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss / 1

        loss.backward()

        if iter_index % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        inputs, labels = prefetcher.next()

        if iter_index % 100 == 0:
            print(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: {scheduler.get_last_lr()[0]:.6f}, loss_total: {loss.item():.6f}"
            )
        iter_index += 1
    scheduler.step()
    return losses.avg


def trainer_logits(model, train_loader, criterion, optimizer, scheduler, epoch):
    start_time = time.time()
    losses = train(train_loader, model, criterion, optimizer, scheduler, epoch)
    # remember best prec@1 and save checkpoint
    end_time = time.time()
    print(
        f"val: epoch {epoch:0>3d}, time consumed: {end_time - start_time}s, loss: {losses:.6f}"
    )
    torch.save(
        {
            'epoch': epoch,
            'loss': losses,
            'lr': scheduler.get_lr()[0],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join('./', 'latest.pth'))


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
