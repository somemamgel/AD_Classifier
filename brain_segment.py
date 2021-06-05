import nibabel as nib
from deepbrain import Extractor
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import random
import torch.nn.functional as f
import torch
import pandas
import re


ext = Extractor()   #  


def trim(arr, mask):
    #
    bounding_box = tuple(
        slice(np.min(i),
              np.max(i) + 1) for i in np.where(mask))
    return arr[bounding_box]


def image_resample(image: sitk.Image):
    '''
    image = sitk.ReadImage(image_path)
    使用 SimpleITK 自带函数重新缩放图像
    '''
    origin_spacing = image.GetSpacing()     #  获取源分辨率
    origin_size = image.GetSize()

    new_spacing = [1, 1, 1]     #  设置新分辨率

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(0)

    resample.SetOutputSpacing(new_spacing)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())

    #  计算新图像的大小
    new_size = [
        int(np.round(origin_size[0] * (origin_spacing[0] / new_spacing[0]))),
        int(np.round(origin_size[1] * (origin_spacing[1] / new_spacing[1]))),
        int(np.round(origin_size[2] * (origin_spacing[2] / new_spacing[2])))
    ]
    resample.SetSize(new_size)

    new_image = resample.Execute(image)
    return new_image


def main():
    '''
    pattern = "*.nii" or pattern = "*.nii.gz"   匹配文件拓展名
    从 data_path 读取文件，处理后保存至 save_path
    '''
    pattern = "*.nii"

    data_path = Path(r"/path/to/adni/")
    save_path = Path(r"/path/to/save")

    file_list = [x for x in data_path.rglob(pattern)]

    for i in file_list:
        data = sitk.ReadImage(str(i))
        data = image_resample(data)

        data = sitk.GetArrayFromImage(data) #  图片中读取 data 为 numpy 数组

        prob = ext.run(data)    #  获取某体素为大脑组织的概率数组
        mask = prob < 0.5       #  从概率数组获取 mask 数组
        data[mask] = 0          #  将非大脑组织区域体素清空
        img = trim(data, data != 0)     #  仅保留含有大脑组织的区域
        # data = f.normalize(torch.from_numpy(data), p=2, dim=-1).numpy()   #  L2 归一化
        data = sitk.GetImageFromArray(data)
        sitk.WriteImage(data, str(save_path / (i.name + '.gz')))  # 压缩存储


def get_shape_info_from_path(data_path: Path, pattern: str):
    '''
    data_path = Path(r"")
    pattern = "*.nii.gz"
    从所给路径获取匹配文件的 shape 信息
    '''
    file_list = [x for x in data_path.rglob(pattern)]

    i_max = [0, 0, 0]
    i_min = [600, 600, 600]

    for i in file_list:
        shape = nib.load(str(i)).get_fdata().shape
        if i_max[0] < shape[0]:
            i_max[0] = shape[0]
        if i_max[1] < shape[1]:
            i_max[1] = shape[1]
        if i_max[2] < shape[2]:
            i_max[2] = shape[2]
        if i_min[0] > shape[0]:
            i_min[0] = shape[0]
        if i_min[1] > shape[1]:
            i_min[1] = shape[1]
        if i_min[2] > shape[2]:
            i_min[2] = shape[2]

    # print(i_max)
    # print(i_min)
    return i_max, i_min


if __name__ == '__main__':
    main()
