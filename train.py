import os
import re
import time
from pathlib import Path

import numpy
import psutil
import torch
import torch.backends.cudnn
import torch.nn.functional as f
from prefetch_generator import BackgroundGenerator
from torch.utils.data.sampler import SubsetRandomSampler

from model import resnet
from my_dataset import MyDataSet


class DataLoaderX(torch.utils.data.DataLoader):  # prefetch_generator，可能会更慢
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
else:
    raise Exception(print("No CUDA device available!"))

batch_size = 8
model_depth = 152
num_epoch = 2000
num_worker = 4
checkpoint_flag = 5
train_flag = True
resume_flag = False

data_path = Path("/content/dataset")
metadata_path = Path("")  # ADNI1_Complete_1Yr_1.5T_3_20_2021.csv
checkpoint_path = Path("")
resume_model_name = Path("")

train_dataset = MyDataSet(images_dir_path=data_path,
                          csv_file_path=metadata_path,
                          is_train=True)
val_dataset = MyDataSet(images_dir_path=data_path,
                        csv_file_path=metadata_path,
                        is_train=False)

train_size = int(0.8 * len(train_dataset))
val_size = int(0.2 * len(train_dataset))

indices = list(range(len(train_dataset)))
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

model = resnet.generate_model(model_depth=model_depth,
                              n_input_channels=1,
                              n_classes=3)

train_loader = DataLoaderX(dataset=train_dataset,
                           sampler=train_sampler,
                           batch_size=batch_size,
                           num_workers=num_worker,
                           drop_last=True)
val_loader = DataLoaderX(dataset=val_dataset,
                         sampler=val_sampler,
                         batch_size=1,
                         num_workers=num_worker)

if resume_flag:
    save_state = torch.load(checkpoint_path / resume_model_name)
    start_epoch = int(re.findall("epoch_([0-9]*)",
                                 str(resume_model_name))[0]) + 1
    model.load_state_dict(save_state['net'])
    model.cuda()
else:
    start_epoch = 1
    model.cuda()

criterion = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(),lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_acc = 0.0
for epoch in range(start_epoch, num_epoch + 1):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    t_num = 0
    if train_flag:
        model.train(True)
        for i, data in enumerate(train_loader):
            images = data[0].cuda()
            labels = data[1].cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            for j, o in enumerate(outputs):
                if torch.argmax(o) == labels[j]:
                    train_acc += 1
            t_num = i + 1
            print("\rTraining. Epoch: {}\t|\ti: {}".format(epoch, i), end="")

    model.train(False)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data[0].cuda()
            labels = data[1].cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            for j, o in enumerate(outputs):
                if torch.argmax(o) == labels[j]:
                    val_acc += 1
            print("\rValidating. Epoch: {}\t|\ti: {}".format(epoch, i), end="")

    res_text = "[{:3d}/{:3d}] {:.2f} sec(s) Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}".format(
        epoch, num_epoch,
        time.time() - epoch_start_time, train_acc / train_size,
        train_loss / t_num, val_acc / val_size, val_loss / val_size)

    print(res_text)

    with open("/content/gdrive/MyDrive/log.txt", "a") as f:
        f.write(res_text)

    if (val_acc / val_size) > best_val_acc:
        best_val_acc = (val_acc / val_size)
        model_save_file_name = "best_model_save.pth"
        torch.save({
            'net': model.state_dict(),
            'epoch': epoch
        }, checkpoint_path / Path(model_save_file_name))

    if epoch % checkpoint_flag == 0:
        model_save_file_name = "resnet_model_save_epoch_{}.pt".format(epoch)
        torch.save({
            'net': model.state_dict(),
            'epoch': epoch
        }, checkpoint_path / Path(model_save_file_name))
