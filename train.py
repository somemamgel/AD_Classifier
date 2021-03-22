import torch
import time
import os
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch: int, data_loader, model, criterion, optimizer, device, current_lr, epoch_logger, batch_logger,
                tb_writer=None, distributed=False):
    print('train at epoch {}'.format(epoch))
