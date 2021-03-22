from pathlib import Path
import nibabel
import pandas
import re
import numpy
import time
from torch.utils.data import Dataset
import skimage.transform


def img_preprocess(data):
    img = numpy.array(data.get_fdata())
    time_start = time.time()
    img = skimage.transform.resize(img, (145, 121, 121))
    time_end = time.time()
    print('Time cost:', time_end - time_start, 's')
    return img


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
        return img, label
