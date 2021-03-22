import os
import numpy
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
from my_dataset import MyDataSet
from model import voxresnet
import torch
import psutil
import torch.backends.cudnn

if __name__ == '__main__':
    if torch.cuda.is_available():
        selected_device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
    else:
        raise Exception(print('No CUDA device available!'))

    batch_size = 2
    model_depth = 50
    num_epoch = 2
    num_worker = 0
    num_checkpoint = 5
    resume_flag = False

    data_path = Path('E:/GraduationDesign/1.5T/ADNI')
    metadata_path = Path('E:/GraduationDesign/1.5T/ADNI1_Complete_1Yr_1.5T_3_20_2021.csv')

    used_dataset = MyDataSet(images_dir_path=data_path, csv_file_path=metadata_path)

    train_size = int(0.7 * len(used_dataset))
    val_size = int(0.1 * len(used_dataset))
    test_size = len(used_dataset) - train_size - val_size

    indices = list(range(len(used_dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    select_model = voxresnet.VoxResNet(1, 3)
    select_model.to(selected_device)

    # used_dataloader = torch.utils.data.DataLoader(dataset=used_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    train_loader = torch.utils.data.DataLoader(dataset=used_dataset, sampler=train_sampler, batch_size=batch_size,
                                               num_workers=num_worker)
    # val_loader = torch.utils.data.DataLoader(dataset=used_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=num_worker)
    # test_loader = torch.utils.data.DataLoader(dataset=used_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_worker)

    criterion = torch.nn.CrossEntropyLoss()

    # ---------
    running_loss = 0.0
    optimizer = torch.optim.SGD(select_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for epoch in range(1, num_epoch + 1):
        select_model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            print(psutil.virtual_memory())
            outputs = select_model(data[0])
            loss = criterion(outputs, data[1])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Train Epoch Done: {}\t Loss: {:.6f}'.format(epoch, loss_mean.item()))
