from tqdm import tqdm
import argparse
import os
import torch
import torch.nn as nn
from dataloader.image_reader import *
from model.DeepLidar import deepLidar
import torch.nn.functional as F
from PIL import Image
from training.utils import *
from env import PREDICTED_RESULT_DIR, KITTI_DATASET_PATH

parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-m', '--model_path', help='loaded model path')
parser.add_argument('-n', '--num_testing_image', type=int, default=10, 
                    help='The number of testing image to be runned')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-s', '--save_fig', action='store_true', help='save predicted result or not')

args = parser.parse_args()



DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'



def rmse(pred, gt):
    dif = gt[np.where(gt>0)] - pred[np.where(gt>0)]
    error = np.sqrt(np.mean(dif**2))
    return error   

def test(model, rgb, lidar, mask):
    model.eval()

    model = model.to(DEVICE)
    rgb = rgb.to(DEVICE)
    lidar = lidar.to(DEVICE)
    mask = mask.to(DEVICE)

    with torch.no_grad():
        color_path_dense, normal_path_dense, color_attn, normal_attn, surface_normal = model(rgb, lidar, mask, stage='A')

        predicted_dense, pred_color_path_dense, pred_normal_path_dense = \
                            get_predicted_depth(color_path_dense, normal_path_dense, color_attn, normal_attn)
 
        
        return torch.squeeze(predicted_dense).cpu()

def get_testing_img_paths():
    gt_folder = os.path.join(KITTI_DATASET_PATH, 'depth_selection', 'val_selection_cropped', 'groundtruth_depth')
    rgb_folder = os.path.join(KITTI_DATASET_PATH, 'depth_selection', 'val_selection_cropped', 'image')
    lidar_folder = os.path.join(KITTI_DATASET_PATH, 'depth_selection', 'val_selection_cropped', 'velodyne_raw')

    gt_filenames = sorted([img for img in os.listdir(gt_folder)])
    rgb_filenames = sorted([img for img in os.listdir(rgb_folder)])
    lidar_filenames = sorted([img for img in os.listdir(lidar_folder)])

    gt_paths = [os.path.join(gt_folder, fn) for fn in gt_filenames]
    rgb_paths = [os.path.join(rgb_folder, fn) for fn in rgb_filenames]
    lidar_paths = [os.path.join(lidar_folder, fn) for fn in lidar_filenames]

    return rgb_paths, lidar_paths, gt_paths

def main():
    # get image paths
    rgb_paths, lidar_paths, gt_paths = get_testing_img_paths()

    # set the number of testing images
    num_testing_image = len(rgb_paths) if args.num_testing_image == -1 else args.num_testing_image

    # load model
    model = deepLidar()
    dic = torch.load(args.model_path, map_location=DEVICE)
    state_dict = dic["state_dict"]
    model.load_state_dict(state_dict)
    print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))


    transformer = image_transforms()
    pbar = tqdm(range(num_testing_image))
    running_error = 0

    for idx in pbar:
        # read image
        print(rgb_paths[idx])
        print(lidar_paths[idx])
        exit()
        rgb = read_rgb(rgb_paths[idx]) # h x w x 3
        lidar, mask = read_lidar(lidar_paths[idx]) # h x w x 1
        gt = read_gt(gt_paths[idx]) # h x w x 1

        # transform numpy to tensor and add batch dimension
        rgb = transformer(rgb).unsqueeze(0)
        lidar, mask = transformer(lidar).unsqueeze(0), transformer(mask).unsqueeze(0)
        
        # saved file path
        fn = os.path.basename(rgb_paths[idx])
        saved_path = os.path.join(PREDICTED_RESULT_DIR, fn)

        # run model
        pred = test(model, rgb, lidar, mask).numpy()
        pred = np.where(pred <= 0.0, 0.9, pred)

        gt = gt.reshape(gt.shape[0], gt.shape[1])
        rmse_loss = rmse(pred, gt)*1000

        running_error += rmse_loss
        mean_error = running_error / (idx + 1)
        pbar.set_description('Mean error: {:.4f}'.format(mean_error))

        if args.save_fig:
            # save image
            pred_show = pred * 256.0
            pred_show = pred_show.astype('uint16')
            res_buffer = pred_show.tobytes()
            img = Image.new("I", pred_show.T.shape)
            img.frombytes(res_buffer, 'raw', "I;16")
            img.save(saved_path)


if __name__ == '__main__':
    main()