import argparse
import torch
from training.utils import *
from PIL import Image
from model.DeepLidar import deepLidar
from dataloader.image_reader import *

parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('--model_path', help='path of pretrained model')
parser.add_argument('--cpu', action='store_true', help='use cpu')
parser.add_argument('--rgb', help='path of rgb image')
parser.add_argument('--lidar', help='path of lidar image')
parser.add_argument('--saved_path', help='path of predicted image')

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'


def main():
    # load model
    model = deepLidar()
    dic = torch.load(args.model_path, map_location=DEVICE)
    state_dict = dic["state_dict"]
    model.load_state_dict(state_dict)
    print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))
    model = model.to(DEVICE)

    # read image
    rgb = read_rgb(args.rgb) # h x w x 3
    lidar, mask = read_lidar(args.lidar)

    # transform numpy to tensor and add batch dimension
    transformer = image_transforms()
    rgb = transformer(rgb).unsqueeze(0).to(DEVICE)
    lidar, mask = transformer(lidar).unsqueeze(0).to(DEVICE), transformer(mask).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        color_path_dense, normal_path_dense, color_attn, normal_attn, surface_normal = model(rgb, lidar, mask, stage='A')
    predicted_dense, pred_color_path_dense, pred_normal_path_dense = \
                        get_predicted_depth(color_path_dense, normal_path_dense, color_attn, normal_attn)

    pred = torch.squeeze(predicted_dense).cpu().numpy()
    pred = np.where(pred <= 0.0, 0.9, pred)


    pred_show = pred * 256.0
    pred_show = pred_show.astype('uint16')
    res_buffer = pred_show.tobytes()
    img = Image.new("I", pred_show.T.shape)
    img.frombytes(res_buffer, 'raw', "I;16")
    img.save(args.saved_path)


if __name__ == '__main__':
    main()