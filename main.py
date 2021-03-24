import os
import numpy
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
from my_dataset import MyDataSet
from model import voxresnet, resnet
import torch
import psutil
import time
import re
import torch.backends.cudnn

if __name__ == '__main__':
    print()
    if torch.cuda.is_available():
        selected_device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        raise Exception(print('No CUDA device available!'))

    batch_size = 1
    model_depth = 50
    num_epoch = 100
    num_worker = 4
    checkpoint_flag = 5
    resume_flag = True

    data_path = Path('E:/GraduationDesign/1.5T/ADNI/')
    metadata_path = Path('E:/GraduationDesign/1.5T/ADNI1_Complete_1Yr_1.5T_3_20_2021.csv')
    checkpoint_path = Path('./checkpoint/')
    resume_model_name = Path('model_save_epoch_10.pt')

    used_dataset = MyDataSet(images_dir_path=data_path, csv_file_path=metadata_path)

    train_size = int(0.8 * len(used_dataset))
    val_size = int(0.2 * len(used_dataset))
    test_size = len(used_dataset) - train_size - val_size

    indices = list(range(len(used_dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # selected_model = voxresnet.VoxResNet(in_channels=1, num_class=3)
    selected_model = resnet.generate_model(model_depth=model_depth, n_input_channels=1, n_classes=3)
    if resume_flag:
        selected_model.load_state_dict(torch.load(checkpoint_path / resume_model_name))
        start_epoch = int(re.findall('epoch_([0-9]*)', str(resume_model_name))[0]) + 1
    else:
        start_epoch = 1
    selected_model.to(selected_device)

    # used_dataloader = torch.utils.data.DataLoader(dataset=used_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    train_loader = torch.utils.data.DataLoader(dataset=used_dataset, sampler=train_sampler, batch_size=batch_size,
                                               num_workers=num_worker, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=used_dataset, sampler=val_sampler, batch_size=batch_size,
                                             num_workers=num_worker)
    test_loader = torch.utils.data.DataLoader(dataset=used_dataset, sampler=test_sampler, batch_size=batch_size,
                                              num_workers=num_worker)

    criterion = torch.nn.CrossEntropyLoss()

    # ---------
    optimizer = torch.optim.SGD(selected_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for epoch in range(start_epoch, num_epoch + 1):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        selected_model.train(True)
        for i, data in enumerate(train_loader):
            print('In training process. Epoch: {}\t|\ti value: {}'.format(epoch, i), end='\r')
            images = data[0].to(selected_device)
            labels = data[1].to(selected_device)
            optimizer.zero_grad()
            # print(psutil.virtual_memory())
            outputs = selected_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            for j, o in enumerate(outputs):
                if torch.argmax(o) == labels[j]:
                    train_acc += 1

        selected_model.train(False)
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                print('In validating process. Epoch: {}\t|\ti value: {}'.format(epoch, i), end='\r')
                images = data[0].to(selected_device)
                labels = data[1].to(selected_device)
                outputs = selected_model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                for j, o in enumerate(outputs):
                    if torch.argmax(o) == labels[j]:
                        val_acc += 1

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % (
            epoch, num_epoch, time.time() - epoch_start_time, train_acc / train_indices.__len__(),
            train_loss / train_indices.__len__(), val_acc / val_indices.__len__(),
            val_loss / val_indices.__len__()))
        with open('./log.txt', 'a') as f:
            f.write('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f\n' % (
                epoch, num_epoch, time.time() - epoch_start_time, train_acc / train_indices.__len__(),
                train_loss / train_indices.__len__(), val_acc / val_indices.__len__(),
                val_loss / val_indices.__len__()))

        if epoch % checkpoint_flag == 0:
            save_file_name = 'model_save_epoch_{}.pt'.format(epoch)
            torch.save(selected_model.state_dict(), checkpoint_path / Path(save_file_name))
