import os
from torch.utils.data import Dataset, DataLoader
from skimage import io
import numpy as np
import random
from dataloader.image_reader import *
from env import KITTI_DATASET_PATH

INTRINSICS = {
    "2011_09_26": (721.5377, 609.5593, 172.8540),
    "2011_09_28": (707.0493, 604.0814, 180.5066),
    "2011_09_29": (718.3351, 600.3891, 181.5122),
    "2011_09_30": (707.0912, 601.8873, 183.1104),
    "2011_10_03": (718.8560, 607.1928, 185.2157),
}

def get_loader(split='train', batch_size=8, shuffle=True, num_workers=8, num_data=None, crop=True):
    """Get torch dataloader."""
    rgb_image_paths, lidar_image_paths, gt_image_paths, normal_image_paths = get_paths(split)
    dataset = depth_dataset(rgb_image_paths, lidar_image_paths, gt_image_paths, normal_image_paths, num_data, crop=crop)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader

class depth_dataset(Dataset):
    """Depth dataset."""

    def __init__(self, rgb_image_paths, lidar_image_paths, gt_image_paths, normal_image_paths=None, num_data=None, h=128, w=256, crop=True):
        """
        Params:
        h: the height of cropped image
        w: the width of cropped image
        """
        self.rgb_image_paths = np.array(rgb_image_paths)
        self.lidar_image_paths = np.array(lidar_image_paths)
        self.gt_image_paths = np.array(gt_image_paths)
        
        if normal_image_paths:
            self.normal_image_paths = np.array(normal_image_paths)

        if num_data:
            np.random.seed(0)
            indices = np.random.choice(len(rgb_image_paths), size=num_data, replace=False)
            self.rgb_image_paths = self.rgb_image_paths[indices]
            self.lidar_image_paths = self.lidar_image_paths[indices]
            self.gt_image_paths = self.gt_image_paths[indices]
            if normal_image_paths:
                self.normal_image_paths = self.normal_image_paths[indices]
            
        self.crop_or_not = crop
        self.h = h
        self.w = w
    
        self.transforms = image_transforms()

    def __len__(self):
        return len(self.rgb_image_paths)

    def __getitem__(self, idx):
        """
        Returns:
        rgb: 3 x 128 x 256
        lidar: 1 x 128 x 256
        mask: 1 x 128 x 256
        gt: 1 x 128 x 256
        params: 128 x 256 x 3
        surface_normal: 3 x 128 x 256
        mask_normal: 3 x 128 x 256
        """
        date = self.rgb_image_paths[idx].split('/')[6][:10] # use to get intrinsic

        intrinsics = INTRINSICS[date]
        params = np.ones((self.h, self.w, 3)).astype('float32')
        params[:, :, 0] = params[:, :, 0] * intrinsics[0]
        params[:, :, 1] = params[:, :, 1] * intrinsics[1]
        params[:, :, 2] = params[:, :, 2] * intrinsics[2]

        # read image
        rgb = read_rgb(self.rgb_image_paths[idx])
        lidar, mask = read_lidar(self.lidar_image_paths[idx])
        gt = read_gt(self.gt_image_paths[idx])
        surface_normal, mask_normal = read_normal(self.normal_image_paths[idx])

        # random crop images
        height, width, channel = rgb.shape

        x_lefttop = random.randint(0, height - self.h)
        y_lefttop = random.randint(0, width - self.w)

        if self.crop_or_not:
            rgb = self._crop(rgb, x_lefttop, y_lefttop, self.h, self.w)
            lidar = self._crop(lidar, x_lefttop, y_lefttop, self.h, self.w)
            mask = self._crop(mask, x_lefttop, y_lefttop, self.h, self.w)
            gt = self._crop(gt, x_lefttop, y_lefttop, self.h, self.w)
            surface_normal = self._crop(surface_normal, x_lefttop, y_lefttop, self.h, self.w)
            mask_normal = self._crop(mask_normal, x_lefttop, y_lefttop, self.h, self.w)

        return self.transforms(rgb), self.transforms(lidar), self.transforms(mask), self.transforms(gt), self.transforms(params),\
            self.transforms(surface_normal), self.transforms(mask_normal)
        
        #return self.transforms(rgb), self.transforms(lidar), self.transforms(mask), self.transforms(gt), self.transforms(params)
        

    def _crop(self, img, x, y, h, w):
        """Crop image
        
        Params:
        img: np array with shape (height, width, channel)
        x: left top point of x-coord
        y: left top point of y-coord
        h: height of cropped image
        w: width of cropped image
        
        Returns:
        img: cropped image with shape (h, w, channel)
        """
        return img[x:x+h, y:y+w, :]
# containing data_depth_annotated, data_depth_rgb, data_depth_velodyne

rgb_folder = os.path.join(KITTI_DATASET_PATH, 'data_depth_rgb')
lidar_folder = os.path.join(KITTI_DATASET_PATH, 'data_depth_velodyne')
gt_folder = os.path.join(KITTI_DATASET_PATH, 'data_depth_annotated')
normal_folder = os.path.join(KITTI_DATASET_PATH, 'data_depth_normals')

rgb2_subfolder = os.path.join('image_02', 'data')
rgb3_subfolder = os.path.join('image_03', 'data')

lidar2_subfolder = os.path.join('proj_depth', 'velodyne_raw', 'image_02')
lidar3_subfolder = os.path.join('proj_depth', 'velodyne_raw', 'image_03')

gt2_subfolder = os.path.join('proj_depth', 'groundtruth', 'image_02') # the same as normal
gt3_subfolder = os.path.join('proj_depth', 'groundtruth', 'image_03')

def get_paths(split='train'):
    """Get all the paths of rgb images, lidar images, and groundtruth images

    Params:
    split: train or val

    Returns:
    rgb_image_paths: list of path of rgb images
    lidar_image_paths: list of path of lidar images
    gt_image_paths: list of path of gt images
    """
    assert split in {'train', 'val'}

    # list of 2011_09_28_drive_0128_sync ...
    date_folder_list = os.listdir(os.path.join(lidar_folder, split))
    date_folder_list.sort()

    rgb_image_paths = []
    lidar_image_paths = []
    gt_image_paths = []
    normal_image_paths = []

    for date_folder in date_folder_list:
        # set base dir of rgb, lidar, gt, normal
        rgb2_base = os.path.join(rgb_folder, split, date_folder, rgb2_subfolder)
        rgb3_base = os.path.join(rgb_folder, split, date_folder, rgb3_subfolder)

        lidar2_base = os.path.join(lidar_folder, split, date_folder, lidar2_subfolder)
        lidar3_base = os.path.join(lidar_folder, split, date_folder, lidar3_subfolder)   
        
        gt2_base = os.path.join(gt_folder, split, date_folder, gt2_subfolder)
        gt3_base = os.path.join(gt_folder, split, date_folder, gt3_subfolder)         
        
        normal2_base = os.path.join(normal_folder, split, date_folder, gt2_subfolder)
        normal3_base = os.path.join(normal_folder, split, date_folder, gt3_subfolder) 

        # the whole image filename in the directory
        img_filenames = os.listdir(os.path.join(lidar_folder, split, date_folder, lidar2_subfolder))
        img_filenames.sort()

        # store whole path to image
        rgb_image_paths.extend([os.path.join(rgb2_base, fn) for fn in img_filenames])
        rgb_image_paths.extend([os.path.join(rgb3_base, fn) for fn in img_filenames])

        lidar_image_paths.extend([os.path.join(lidar2_base, fn) for fn in img_filenames])
        lidar_image_paths.extend([os.path.join(lidar3_base, fn) for fn in img_filenames])

        gt_image_paths.extend([os.path.join(gt2_base, fn) for fn in img_filenames])
        gt_image_paths.extend([os.path.join(gt3_base, fn) for fn in img_filenames])

        normal_image_paths.extend([os.path.join(normal2_base, fn) for fn in img_filenames])
        normal_image_paths.extend([os.path.join(normal3_base, fn) for fn in img_filenames])

    assert len(rgb_image_paths) == len(lidar_image_paths) == len(gt_image_paths) == len(normal_image_paths)

    print('The number of {} data: {}'.format(split, len(rgb_image_paths)))

    return rgb_image_paths, lidar_image_paths, gt_image_paths, normal_image_paths


if __name__ == '__main__':
    get_paths()
    loader = get_loader('train')
    for rgb, lidar, mask, gt, params in loader:
        pass