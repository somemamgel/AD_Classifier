from pathlib import Path
import nibabel
import pandas
import re
import numpy
import torch
import random
from torch.utils.data import Dataset
import random
import torch.nn.functional as f


def random_crop_3d(img, crop_size):
    '''
    img: numpy.ndarray
    crop_size: tuple
    从 img 数组随机剪切指定 crop_size 大小的数组
    '''
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    crop_img = img[x_random:x_random + crop_size[0],
                   y_random:y_random + crop_size[1],
                   z_random:z_random + crop_size[2]]

    return crop_img


def img_preprocess(data, is_train: bool):
    '''
    图像预处理，为训练集则随机裁剪缩放
    '''
    img = numpy.array(data.get_data()).astype(numpy.float32)
    if is_train:

        #  随机旋转
        img = numpy.rot90(img, k=int(random.random() * 4), axes=(0, 1))
        img = numpy.rot90(img, k=int(random.random() * 4), axes=(0, 2))
        img = numpy.rot90(img, k=int(random.random() * 4), axes=(1, 2)).copy()

        img = f.normalize(torch.from_numpy(img), p=2, dim=-1)   #  L2 正则化
        img = random_crop_3d(img.numpy(), (110, 110, 110))      #  随机裁剪为 110*110*110
    else:
        img = f.normalize(torch.from_numpy(img), p=2, dim=-1).numpy()
    return img[None, ...]


class MyDataSet(Dataset):
    '''
    重写 __init__(), __len__(), __getitem__() 三个方法来自定义 dataset 类
    '''
    def __init__(self, images_dir_path: Path, csv_file_path: Path, is_train):
        pattern = "*.nii.gz"    #  训练时所用的文件拓展名
        super().__init__()

        #  从 metadata.csv 文件中读取文件名和分类组别的映射关系，并保存到字典中
        self.metadata: dict = {
            i[1]['Subject'] + '-' + i[1]['Image Data ID']: i[1]['Group']
            for i in pandas.read_csv(csv_file_path).iterrows()
        }

        self.file_list: list = [x for x in images_dir_path.rglob(pattern)]  # ('*.nii.gz')
        self.is_train = is_train

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: int):
        f_path: Path = self.file_list[index]
        img = img_preprocess(nibabel.load(str(f_path)), self.is_train)
        # ./ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114556321_S13408_I118671.nii

        #  从文件名获取类别
        f = re.findall("ADNI_(.*)_MR_.*(I[0-9]{1,8})\\.nii", str(f_path))[0]
        label = self.metadata[f[0] + '-' + f[1]]

        if label == "AD":
            label = torch.tensor(0)
        elif label == "MCI":
            label = torch.tensor(1)
        elif label == "CN":
            label = torch.tensor(2)
        else:
            raise ValueError(r"Not in AD or NC or MCI")
        return img, label
