from pathlib import Path

import nibabel
import numpy
import pandas
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from model import densenet, resnet
from my_dataset import MyDataSet

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
else:
    raise Exception(print("No CUDA device available!"))

data_path = Path(r"/path/to/data")
metadata_path = Path(r"/path/to/metadata")
model_path = Path(r"/path/to/model")

dataset = MyDataSet(images_dir_path=data_path,
                    csv_file_path=metadata_path,
                    is_train=False)

train_size = int(0.8 * len(dataset))
val_size = int(0.2 * len(dataset))

indices = list(range(len(dataset)))
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
val_sampler = SubsetRandomSampler(val_indices)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    sampler=val_sampler,
)
# model = resnet.generate_model(model_depth=152, n_input_channels=1, n_classes=3)
model = densenet.generate_model(model_depth=121, n_input_channels=1, num_classes=3)

model.load_state_dict(torch.load("./best_resnet_model_save_3_classes.pt")['net'])
model.cuda().eval()

#  混淆矩阵
result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


with torch.no_grad():
    for i, data in enumerate(dataloader):
        images = data[0].cuda()
        labels = data[1].cuda()
        outputs = model(images)
        # print(outputs, torch.argmax(outputs))
        # print(labels + torch.argmax(outputs) * 3)
        print("\r{}/{}".format(i, val_size), end='')
        print(result, end='')
        result[labels][torch.argmax(outputs)] += 1

print(result)

