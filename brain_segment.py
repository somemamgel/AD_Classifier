import nibabel as nib
from deepbrain import Extractor
from pathlib import Path
import matplotlib.pyplot as plt
import numpy
import SimpleITK as sitk
import random
import torch.nn.functional as f
import torch
import pandas
import re

ext = Extractor()


def trim(arr, mask):
    bounding_box = tuple(
        slice(numpy.min(indexes), numpy.max(indexes) + 1)
        for indexes in numpy.where(mask))
    return arr[bounding_box]


def convert(img_path):
    data = nib.load(img_path).get_fdata()
    prob = ext.run(data)
    mask = prob < 0.2
    data[mask] = 0
    data = trim(data, data != 0)
    return nib.Nifti1Image(data, numpy.eye(4))


def img_resample(image):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(0)

    new_spacing = [1, 1, 1]
    resample.SetOutputSpacing(new_spacing)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())

    size = [
        int(numpy.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(numpy.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(numpy.round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    resample.SetSize(size)

    newimage = resample.Execute(image)
    return newimage


def main():
    data_path = Path("")
    save_path = Path("")
    file_list = [x for x in data_path.rglob('*.nii')]

    for i in file_list:
        print(i.name)
        data = sitk.ReadImage(str(i))
        data = img_resample(data)
        data = sitk.GetArrayFromImage(data)
        prob = ext.run(data)
        mask = prob < 0.5
        data[mask] = 0
        data = trim(data, data != 0)
        data = f.normalize(torch.from_numpy(data), p=2, dim=-1).numpy()
        data = sitk.GetImageFromArray(data)
        sitk.WriteImage(data, str(save_path / (i.name + '.gz')))
        pass


def get_shape_info():
    data_path = Path("")
    file_list = [x for x in data_path.rglob("*.nii.gz")]

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
            i_min[1]  = shape[1]
        if i_min[2] > shape[2]:
            i_min[2] = shape[2]

    print(i_max)
    print(i_min)

if __name__ == '__main__':
    main()
