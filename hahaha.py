import time
import torch
from options.train_options import TrainOptions
from models import create_model
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    AddChanneld,
    Compose,
    NormalizeIntensityd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    ToTensord,
    LoadImaged,
    RandSpatialCropd,
    RandAdjustContrastd,
    CropForegroundd,
    RandZoomd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandBiasFieldd,
    RandShiftIntensityd
)
import os
import gzip

from src.util import util


def decompress_nifti_gz_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
            os.makedirs(output_folder) # 确保输出文件夹存在
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename[:-3])
            decompress_nifti_gz(input_file, output_file)
def decompress_nifti_gz(nifti_gz_file, output_nifti_file):
    with gzip.open(nifti_gz_file, 'rb') as gz_file:
        data = gz_file.read()
    with open(output_nifti_file, 'wb') as nifti_file:
        nifti_file.write(data)
'''
def util(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0, :, :, 16].cpu().float().numpy()  # convert it into a numpy array
        # print(image_numpy.shape)
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
        image_numpy -= np.min(image_numpy)
        image_numpy = image_numpy / np.max(image_numpy) * 255.

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
'''
if __name__ == '__main__':
    sct_folder_path = '/media/wuyi/D6DA272ADA2705F9/zfc/yixue/SynCT_TcMRgFUS-main/src/datasets/results'  # 替换为实际的MR文件夹路径
    ct_folder_path = '/media/wuyi/D6DA272ADA2705F9/zfc/yixue/SynCT_TcMRgFUS-main/src/datasets/ct'  # 替换为实际的CT文件夹路径
    sct_output_folder = '/media/wuyi/D6DA272ADA2705F9/zfc/yixue/SynCT_TcMRgFUS-main/src/datasets/sct'
    ct_output_folder = '/media/wuyi/D6DA272ADA2705F9/zfc/yixue/SynCT_TcMRgFUS-main/src/datasets/test_rct'
    '''
    decompress_nifti_gz_folder(sct_folder_path, sct_output_folder)
    decompress_nifti_gz_folder(ct_folder_path, ct_output_folder)
    '''
    sct_paths = sorted(glob.glob(sct_output_folder + '/*'))
    ct_paths = sorted(glob.glob(ct_output_folder + '/*'))
    ct_files = ct_paths[-80:]