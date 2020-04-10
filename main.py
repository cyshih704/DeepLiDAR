import argparse
import os

import torch
import torch.optim as optim
from tqdm import tqdm

from dataloader.dataloader import get_loader
from model.DeepLidar import deepLidar
from tb_writer import TensorboardWriter
from training.train import EarlyStop, train_val
from training.utils import get_depth_and_normal
from env import SAVED_MODEL_PATH

parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('-m', '--saved_model_name', type=str, default='model', help='saved_model_name')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-l', '--load_model', help='load model')
parser.add_argument('-n', '--num_data', type=int, default=20000, help='the number of data used to train')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'


def main_train(model, stage):
    # setting tensorboard
    tensorboard_path = 'runs/{}_{}'.format(stage, args.saved_model_name)
    tb_writer = TensorboardWriter(tensorboard_path)
    
    # get one testing image, used to visualize result in each eopch 
    testing_rgb, testing_lidar, testing_mask = tb_writer.get_testing_img()
    testing_rgb, testing_lidar, testing_mask = testing_rgb.to(DEVICE), testing_lidar.to(DEVICE), testing_mask.to(DEVICE)

    # setting early stop, if result doen't improve more than PATIENCE times, stop iteration
    early_stop = EarlyStop(patience=10, mode='min')

    # get data loader
    loader = {'train': get_loader('train', num_data=args.num_data), \
              'val': get_loader('val', shuffle=False, num_data=1000)}
    

    for epoch in range(args.epoch):
        saved_model_path = os.path.join(SAVED_MODEL_PATH, "{}_{}_e{}".format(args.saved_model_name, stage, epoch+1))
        train_losses, val_losses = train_val(model, loader, epoch, DEVICE, stage)

        # predict dense and surface normal using testing image and write them to tensorboard
        predicted_dense, pred_surface_normal = get_depth_and_normal(model, testing_rgb, testing_lidar, testing_mask)
        tb_writer.tensorboard_write(epoch, train_losses, val_losses, predicted_dense, pred_surface_normal)

        if early_stop.stop(val_losses[0], model, epoch+1, saved_model_path):
            break

    tb_writer.close()




def main():
    model = deepLidar().to(DEVICE)
    if args.load_model:
        dic = torch.load(args.load_model)
        state_dict = dic["state_dict"]
        model.load_state_dict(state_dict)
        print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))

    #main_train(model, 'N')
    #main_train(model, 'D')
    main_train(model, 'A')


if __name__ == '__main__':
    main()
