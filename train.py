import time
import torch
from options.train_options import TrainOptions
from models import create_model
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
from util.visualizer import Visualizer
from glob import glob
import numpy as np
import os
import glob
import nibabel as nib
from PIL import Image

def convert_nifti_to_image(nifti_file, output_file):
    # 加载NIfTI图像
    nifti_img = nib.load(nifti_file)
    data = nifti_img.get_fdata()

    # 将数据缩放到0-255的范围，并转换为无符号整型
    data_scaled = (data - data.min()) / (data.max() - data.min()) * 255
    data_scaled = data_scaled.astype('uint8')

    # 创建图像对象
    image = Image.fromarray(data_scaled)

    # 保存为图像文件
    image.save(output_file)

# 示例用法
import os
import gzip

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
def process_tensor_list_by_8(tensor_list, input):
    processed_list = []
    i = 0
    while i <= 1:
        print(input[i])
        if tensor_list[i][2] % 8 == 0:
            processed_tensor = input[i]
        else:
            processed_tensor = torch.cat((input[i], torch.zeros(tensor_list[i][0], tensor_list[i][1], 8 - (tensor_list[i][2] % 8), tensor_list[i][3], tensor_list[i][4])), dim=2)
        if tensor_list[i][3] % 8 == 0:
            pass
        else:
            processed_tensor = torch.cat((processed_tensor, torch.zeros(tensor_list[i][0], tensor_list[i][1], list(processed_tensor.shape)[2], 8 - (tensor_list[i][3] % 8), tensor_list[i][4])), dim=3)
        processed_list.append(processed_tensor)
        i += 1
    return processed_list

if __name__ == '__main__':
    # 指定文件夹路径
    '''
    mr_folder_path = '/media/wuyi/D6DA272ADA2705F9/zfc/yixue/SynCT_TcMRgFUS-main/src/datasets/mr'  # 替换为实际的MR文件夹路径
    ct_folder_path = '/media/wuyi/D6DA272ADA2705F9/zfc/yixue/SynCT_TcMRgFUS-main/src/datasets/ct'  # 替换为实际的CT文件夹路径
    mr_input_folder = mr_folder_path
    ct_input_folder = ct_folder_path
    mr_output_folder = '/media/wuyi/D6DA272ADA2705F9/zfc/yixue/SynCT_TcMRgFUS-main/src/datasets/mrmr'
    ct_output_folder = '/media/wuyi/D6DA272ADA2705F9/zfc/yixue/SynCT_TcMRgFUS-main/src/datasets/ctct'
    decompress_nifti_gz_folder(mr_input_folder, mr_output_folder)
    decompress_nifti_gz_folder(ct_input_folder, ct_output_folder)
    '''
    mr_paths = sorted(glob.glob('/media/wuyi/D6DA272ADA2705F9/zfc/yixue/synct/src/datasets/mrmr'+'/*.nii'))
    ct_paths = sorted(glob.glob('/media/wuyi/D6DA272ADA2705F9/zfc/yixue/synct/src/datasets/ctct' + '/*.nii'))
     # 创建data_dicts列表
    # data_dicts = []
   #
    data_dicts = [{
            "A": mr_path,
            "B": ct_path,
            'A_paths': mr_path,
            'B_paths': ct_path
        } for mr_path, ct_path in zip(mr_paths, ct_paths)]
       #  data_dicts.append(data_dict)

 # 取前10个文件作为训练文件
    train_files = data_dicts[:100]

    # 进行数据转换
    trainTransform = Compose([
        LoadImaged(keys=["A", "B"]),
        AddChanneld(keys=["A", "B"]),
        # 其他的转换操作...
        # MRI pre-processing
        NormalizeIntensityd(keys="A", nonzero=True),  # z-score normalization
        ScaleIntensityRangePercentilesd(keys="A", lower=0.01, upper=99.9, b_min=-1.0, b_max=1.0, clip=True,
                                        relative=False),  # normalize the intensity to [-1, 1]

        # CT pre-processing
        ScaleIntensityRanged(keys="B", a_min=-1024, a_max=3071, b_min=-1.0, b_max=1.0, clip=True),

        # Spatial augmentation
        # RandAffined(keys=["A", "B"],
        #             prob=0.2,
        #             rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        #             mode=('bilinear', 'bilinear')
        #             ),

        # crop 256 x 256 x 32 volumes
        RandSpatialCropd(keys=["A", "B"], roi_size=(256, 256, 32), random_size=False),
        # randomly crop patches of 256 x 256 x 32

        # Intensity augmentation
        RandShiftIntensityd(keys="A", offsets=(-0.1, 0.1), prob=0.2),
        RandAdjustContrastd(keys="A", prob=0.2, gamma=(0.8, 1.2)),
        ToTensord(keys=["A", "B"])])

    train_ds = Dataset(data=train_files, transform=trainTransform) ##mr_output_folder
    train_loader = DataLoader(train_ds,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0)

    opt = TrainOptions().parse()  # get training options
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_loader)  # get the number of images in the dataset.
    # print('The number of training images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    best_epoch = 0
    best_metric = float('inf')
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_loader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            value1 = data["A"]
            print(value1.shape)
            value2 = data["B"]
            print(value2.shape)
            inputs = [value1, value2]
            input_shapes = []
            for input in inputs:
                input_shape = list(input.shape)
                input_shapes.append(input_shape)
            output_list = process_tensor_list_by_8(input_shapes, inputs)
            for tensor in output_list:
                print(tensor.shape)
            data['A'] = output_list[0]
            data['B'] = output_list[1]
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters() # calculate loss functions, get gradients, update network weights



            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        metric = model.loss_G
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            model.save_networks('best')

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
