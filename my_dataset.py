from pathlib import Path
import nibabel
import pandas
import re
import numpy
import torch

from torch import float64
from torch.utils.data import Dataset
import skimage.transform


def img_preprocess(data):
    img = skimage.transform.resize(numpy.array(data.get_data()), (145, 145, 145))
    # img = numpy.array(data.get_data()).astype(numpy.float32)
    # channel = numpy.array([1])
    # time_start = time.time()
    return img[None, ...]
    # time_end = time.time()
    # print('Time cost:', time_end - time_start, 's')
    # return img


class MyDataSet(Dataset):
    def __init__(self, images_dir_path: Path, csv_file_path: Path):
        super().__init__()
        self.metadata: dict = {i[1]['Subject'] + '-' + i[1]['Image Data ID']: i[1]['Group'] for i in
                               pandas.read_csv(csv_file_path).iterrows()}
        self.file_list: list = [x for x in images_dir_path.rglob('*.nii')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: int):
        f_path: Path = self.file_list[index]
        img = img_preprocess(nibabel.load(str(f_path)))
        # ./ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114556321_S13408_I118671.nii
        f = re.findall('ADNI_(.*)_MR_.*(I[0-9]{1,8})\\.nii', str(f_path))[0]
        label = self.metadata[f[0] + '-' + f[1]]
        if label == 'AD':
            # label = torch.tensor([1.0, 0.0, 0.0])
            label = torch.tensor(0)
        elif label == 'MCI':
            # label = torch.tensor([0.0, 1.0, 0.0])
            label = torch.tensor(1)
        elif label == 'CN':
            # label = torch.tensor([0.0, 0.0, 1.0])
            label = torch.tensor(2)
        return img, label
