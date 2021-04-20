import os
import numpy
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
from prefetch_generator import BackgroundGenerator
from my_dataset import MyDataSet
from model import resnet
import torch
import psutil
import time
import re
import torch.backends.cudnn
import torch.nn.functional as f


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    raise Exception(print("No CUDA device available!"))

batch_size = 8
model_depth = 152
num_epoch = 2000
num_worker = 4
checkpoint_flag = 5
train_flag = True
resume_flag = True

data_path = Path("/content/dataset")
metadata_path = Path("/content/gdrive/MyDrive/metadata.csv")
checkpoint_path = Path("/content/gdrive/MyDrive/Result/")
resume_model_name = Path("resnet_model_save_epoch_1015.pt")

train_dataset = MyDataSet(images_dir_path=data_path, csv_file_path=metadata_path, is_train=True)
test_dataset = MyDataSet(images_dir_path=data_path, csv_file_path=metadata_path, is_train=False)

train_size = int(0.8 * len(train_dataset))
val_size = int(0.2 * len(train_dataset))

indices = list(range(len(train_dataset)))
train_indices = indices[:train_size]
val_indices = indices[train_size: train_size + val_size]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# model = voxresnet.VoxResNet(in_channels=1, num_class=3)
model = resnet.generate_model(model_depth=model_depth, n_input_channels=1, n_classes=3)
# model = resnet.ResNet()

train_loader = DataLoaderX(dataset=train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_worker, drop_last=True)
val_loader = DataLoaderX(dataset=test_dataset, sampler=val_sampler, batch_size=1, num_workers=num_worker)

if resume_flag:
    save_state = torch.load(checkpoint_path / resume_model_name)
    start_epoch = int(re.findall("epoch_([0-9]*)", str(resume_model_name))[0]) + 1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    model.load_state_dict(save_state['net'])
    # optimizer.load_state_dict(save_state['optimizer'])
    model.to(device)
else:
    start_epoch = 1
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=start_epoch-2)

criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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
            images = data[0].to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()
            # print(psutil.virtual_memory())
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            for j, o in enumerate(outputs):
                if torch.argmax(o) == labels[j]:
                    train_acc += 1
            t_num = i
            print("\rIn training process. Epoch: {}\t|\ti value: {}".format(epoch, i), end="")

    model.train(False)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data[0].to(device)
            labels = data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            for j, o in enumerate(outputs):
                if torch.argmax(o) == labels[j]:
                    val_acc += 1
            print("\rIn validating process. Epoch: {}\t|\ti value: {} {}".format(epoch, i, outputs), end="")


    print("[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f" % (
            epoch,
            num_epoch,
            time.time() - epoch_start_time,
            train_acc / train_size,
            train_loss / t_num,
            val_acc / val_size,
            val_loss / val_size,
        )
    )

    with open("/content/gdrive/MyDrive/Result/log.txt", "a") as f:
        f.write(
            "[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f\n"
            % (
                epoch,
                num_epoch,
                time.time() - epoch_start_time,
                train_acc / train_size,
                train_loss / t_num,
                val_acc / val_size,
                val_loss / val_size,
            )
        )
    
    if (val_acc/val_size) > best_val_acc:
        best_val_acc = (val_acc/val_size)
        model_save_file_name = "best_resnet_model_save_epoch.pt"
        torch.save({'net': model.state_dict(), 'epoch': epoch}, checkpoint_path / Path(model_save_file_name))

    if epoch % checkpoint_flag == 0:
        model_save_file_name = "resnet_model_save_epoch_{}.pt".format(epoch)
        torch.save({'net': model.state_dict(), """'optimizer': optimizer.state_dict,""" 'epoch': epoch}, checkpoint_path / Path(model_save_file_name))
    
    # scheduler.step()
