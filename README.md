# DeepLiDAR (Python3, Pytorch 1.4.0)
This repository is the implementation for [DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qiu_DeepLiDAR_Deep_Surface_Normal_Guided_Depth_Prediction_for_Outdoor_Scene_CVPR_2019_paper.pdf). There are some difference between author's repo and mine.

(1) Rewrite the code referenced from [author's repo](https://github.com/JiaxiongQ/DeepLiDAR) with python3.6 and newest version of pytorch. 

(2) Clarify the structure of KITTI depth completion data

(3) Make it easier to reproduce my result (**not author's result**, because by considering computational resources and time, there are some differences between author's implementation and mine)
 - Reduce model parameters from 143,981,012 to 47,750,544. (Original model is too large to put onto single GPU, author used 3 GeForce GTX 1080 Ti GPUs)
 - Smaller training image (128 x 256). (Author uses 256 x 512)
 - Use less data to train (20000 images). (Author used 85898 images, and it takes about 3 days to train)
 - Slight implementation difference 

(4) I add comments on the code and make it more flexible and readable. 

(5) I add tensorboard visualization for every epoch



## Requirements
* Python 3.6.8
```
pip3 install -r requirements.txt
```

## Data Preparation
- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset from their website.
- Use the following scripts to extract corresponding RGB images from the raw dataset. 
```bash
./download/rgb_train_downloader.sh
./download/rgb_val_downloader.sh
```
* The overall code, data, and results directory is structured as follows
  * data_depth_annotated: ground truth data (dense depth)
  * data_depth_velodyne: sparse data (LiDAR)
  * data_rgb: RGB image
  * data_depth_annotated: Used to train surface normals, and generated from **data_depth_annotated** by using **generate_normals.py**

```
.
├── KITTI_data
|   ├── data_depth_annotated
|   |   ├── train
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |       ├── proj_depth
|   |   |           ├── groundtruch
|   |   |               ├── image02
|   |   |               ├── image03
|   |   ├── val
|   |   |   ├── (the same as train)
|   ├── data_depth_velodyne
|   |   ├── train
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |       ├── proj_depth
|   |   |           ├── velodyne_raw
|   |   |               ├── image_02
|   |   |               ├── image_03
|   |   ├── val
|   |   |   ├── (the same as train)
|   ├── data_rgb
|   |   ├── train
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |       ├── image_02
|   |   |       ├── image_03
|   |   ├── val
|   |   |   ├── (the same as train)
|   ├── data_depth_normals
|   |   ├── (the same as data_depth_annotated)
|   └── depth_selection
|   |   ├── test_depth_completion_anonymous
|   |   ├── test_depth_prediction_anonymous
|   |   ├── val_selection_cropped

```

### Path setting
Please set the path in the **env.py** first

```
SAVED_MODEL_PATH = './saved_model' # save model in this directory
KITTI_DATASET_PATH = /PATH/TO/KITTI_data/ # path to KITTI_data as structured in the above
PREDICTED_RESULT_DIR = './predicted_dense' # path to save predicted figures (used in test.py)
```


### To generate **data_depth_normals**
First, enter **surface-normal/** to build and install library. 

Second, run the following script to generate **data_depth_normals**
```
python3 generate_normals.py
```

## Pretrained Model Download
See the next section

## Usage

### Train and validation
```
python3 main.py -b <BATCH_SIZE> -e <EPOCH> -m <SAVED_MODEL_NAME> -l <MODEL_PATH> -n <NUM_DATA> -cpu
    -b <BATCH_SIZE>
        batch size used for training and validation
    -e <EPOCH>
        the number of epoch for training and validation
    -m <SAVED_MODEL_NAME>
        the model name (be saved in SAVED_MODEL_PATH)
    -l <MODEL_PATH>
        specified the model path if you want to load previous model
    -n <NUM_DATA>
        the number of data used for training. (set -1 if you want to use all the training data (85898))
    -cpu
        if you want to use CPU to train
```
There are three different stages of training model.
1. (N) Train surface normal
2. (D) Train depth of color pathway and normal pathway
3. (A) Train the whole network (fix surface normal network)

We test the model with 3 different settings

(A) Train N stage for 15 epoch, train D stage for 15 epoch, and then train A stage for 15 epoch ([download](https://drive.google.com/open?id=1q5crzuMye55SwNMMMY5BDc67M4pziGUM))

(B) Train A for 12 epochs (due to early stop with patience 10, no update parameter of deepLidar.normal (random)) ([download](https://drive.google.com/open?id=1uG6p4wuD9CumYz7hhlCOzkKs7Aoeo6GK))

(c) Train A for 10 epochs (update parameter of deepLidar.normal) ([download](https://drive.google.com/open?id=1Mgf1GfryuwS-NIigqSvg0Uxf0JcvuKdr))


### Test
Test on **depth_selection/val_selection_cropped** data
```
python3 test.py -m <MODEL_PATH> -n <NUM_DATA> -cpu
    -n <NUM_DATA>
        the number of data used for testing. (set -1 if you want to use all the testing data (1000))
    -m <MODEL_PATH>
        the path of loaded model
    -cpu
        if you want to use CPU to test
    -s
        if you want to save predicted figure in PREDICTED_RESULT_DIR
```
The following results are testing on **depth_selection/val_selection_cropped** data
|  Setting   | RMSE (mm)  |
|  ----  | ----  |
| A ([download](https://drive.google.com/open?id=1q5crzuMye55SwNMMMY5BDc67M4pziGUM)) | 1191.6127 |
| B ([download](https://drive.google.com/open?id=1uG6p4wuD9CumYz7hhlCOzkKs7Aoeo6GK)) | 1182.6613 |
| C ([download](https://drive.google.com/open?id=1Mgf1GfryuwS-NIigqSvg0Uxf0JcvuKdr)) | 1026.8722 |

### Test a pair of inputs
Run a pair of rgb and lidar image as input, and then save the predicted dense depth
```
python3 test_a_pair.py --model_path </PATH/TO/PRETRAIN_MODEL> --rgb <PATH/TO/RGB_IMAGE> --lidar <PATH/TO/LIDAR_IMAGE>
                       --saved_path </SAVED_FIGURE/PATH>
    --model_path <MODEL_PATH>
        the path of pretrained model  
    --rgb <PATH>
        the path of rgb image
    --lidar <PATH>
        the path of lidar image
    --saved_path <PATH>
        the path of saved image
```


## Tensorboard Visualization
```
tensorboard --logdir runs/
```

## Experiment of setting A
Input data (rgb image, lidar image)
![image](https://github.com/ChingYenShih/EECS-545-Final/blob/master/final/figure/input.png)

Groundtruth (surface normal, dense depth)
![image](https://github.com/ChingYenShih/EECS-545-Final/blob/master/final/figure/gt.png)

Masked predicted result (masked surface normal, masked dense depth)
![image](https://github.com/ChingYenShih/EECS-545-Final/blob/master/final/figure/mask_pred.png)

Predicted result (surface normal, dense depth)
![image](https://github.com/ChingYenShih/EECS-545-Final/blob/master/final/figure/pred.png)


## Citation 
If you use this method in your work, please cite the following:
```
@InProceedings{Qiu_2019_CVPR,
author = {Qiu, Jiaxiong and Cui, Zhaopeng and Zhang, Yinda and Zhang, Xingdi and Liu, Shuaicheng and Zeng, Bing and Pollefeys, Marc},
title = {DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene From Sparse LiDAR Data and Single Color Image},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
