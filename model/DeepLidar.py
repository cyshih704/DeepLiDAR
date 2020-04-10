import torch
import torch.nn as nn
from model.module import *


class deepCompletionUnit(nn.Module):
    def __init__(self, mode):
        super(deepCompletionUnit, self).__init__()
        assert mode in {'N', 'C', 'I'}
        self.mode = mode

        #s_filter = [32, 99, 195, 387, 515, 512]
        #r_filter = [32, 64, 128, 256, 256, 512]
        r_filter = [32, 64, 128, 128, 256, 256]
        d_filter = [16, 32, 64, 128]
        #s_filter = [32, r_filter[1]+d_filter[0]+3, r_filter[2]+d_filter[1]+3, r_filter[3]+d_filter[2]+3, r_filter[4]+d_filter[3]+3, 256]
        if mode == 'I':
            s_filter = [32, r_filter[1]+d_filter[0]+3, r_filter[2]+d_filter[1]+3, r_filter[3]+d_filter[2]+3, r_filter[4]+d_filter[3]+3, 256]

            self.upsample4 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
            self.upsample3 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
            self.upsample2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
            self.upsample1 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)

            self.predict_normal5 = nn.Conv2d(s_filter[5], 3, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal4 = nn.Conv2d(s_filter[4], 3, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal3 = nn.Conv2d(s_filter[3], 3, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal2 = nn.Conv2d(s_filter[2], 3, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal1 = nn.Conv2d(s_filter[1], 3, kernel_size=3, stride=1, padding=1, bias=True)

            self.conv_sparse1 = ResBlock(channels_in=2, num_filters=s_filter[0], stride=1)
            self.conv_sparse2 = ResBlock(channels_in=s_filter[0], num_filters=s_filter[1], stride=2)
            self.conv_sparse3 = ResBlock(channels_in=s_filter[1], num_filters=s_filter[2], stride=2)
            self.conv_sparse4 = ResBlock(channels_in=s_filter[2], num_filters=s_filter[3], stride=2)
            self.conv_sparse5 = ResBlock(channels_in=s_filter[3], num_filters=s_filter[4], stride=2)
            self.conv_sparse6 = ResBlock(channels_in=s_filter[4], num_filters=s_filter[5], stride=2)

            self.conv_rgb1 = ResBlock(channels_in=3, num_filters=r_filter[0], stride=1)
            self.conv_rgb2 = ResBlock(channels_in=r_filter[0], num_filters=r_filter[1], stride=2)
            self.conv_rgb3 = ResBlock(channels_in=r_filter[1], num_filters=r_filter[2], stride=2)
            self.conv_rgb3_1 = ResBlock(channels_in=r_filter[2], num_filters=r_filter[2], stride=1)
            self.conv_rgb4 = ResBlock(channels_in=r_filter[2], num_filters=r_filter[3], stride=2)
            self.conv_rgb4_1 = ResBlock(channels_in=r_filter[3], num_filters=r_filter[3], stride=1)
            self.conv_rgb5 = ResBlock(channels_in=r_filter[3], num_filters=r_filter[4], stride=2)
            self.conv_rgb5_1 = ResBlock(channels_in=r_filter[4], num_filters=r_filter[4], stride=1)
            self.conv_rgb6 = ResBlock(channels_in=r_filter[4], num_filters=r_filter[5], stride=2)
            self.conv_rgb6_1 = ResBlock(channels_in=r_filter[5], num_filters=r_filter[5], stride=1)

        elif mode == 'C':
            s_filter = [32, r_filter[1]+d_filter[0]+1, r_filter[2]+d_filter[1]+1, r_filter[3]+d_filter[2]+1, r_filter[4]+d_filter[3]+1, 256]

            self.upsample4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            self.upsample3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            self.upsample2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            self.upsample1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)   

            self.predict_normal5 = nn.Conv2d(s_filter[5], 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal4 = nn.Conv2d(s_filter[4], 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal3 = nn.Conv2d(s_filter[3], 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal2 = nn.Conv2d(s_filter[2], 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal1 = nn.Conv2d(s_filter[1], 2, kernel_size=3, stride=1, padding=1, bias=True)

            self.conv_sparse1 = ResBlock(channels_in=2, num_filters=s_filter[0], stride=1)
            self.conv_sparse2 = ResBlock(channels_in=s_filter[0], num_filters=s_filter[1], stride=1)
            self.conv_sparse3 = ResBlock(channels_in=s_filter[1], num_filters=s_filter[2], stride=2)
            self.conv_sparse4 = ResBlock(channels_in=s_filter[2], num_filters=s_filter[3], stride=2)
            self.conv_sparse5 = ResBlock(channels_in=s_filter[3], num_filters=s_filter[4], stride=2)
            self.conv_sparse6 = ResBlock(channels_in=s_filter[4], num_filters=s_filter[5], stride=2)
        
            self.conv_rgb1 = ResBlock(channels_in=3, num_filters=r_filter[0], stride=1)
            self.conv_rgb2 = ResBlock(channels_in=r_filter[0], num_filters=r_filter[1], stride=1)
            self.conv_rgb3 = ResBlock(channels_in=r_filter[1], num_filters=r_filter[2], stride=2)
            self.conv_rgb3_1 = ResBlock(channels_in=r_filter[2], num_filters=r_filter[2], stride=1)
            self.conv_rgb4 = ResBlock(channels_in=r_filter[2], num_filters=r_filter[3], stride=2)
            self.conv_rgb4_1 = ResBlock(channels_in=r_filter[3], num_filters=r_filter[3], stride=1)
            self.conv_rgb5 = ResBlock(channels_in=r_filter[3], num_filters=r_filter[4], stride=2)
            self.conv_rgb5_1 = ResBlock(channels_in=r_filter[4], num_filters=r_filter[4], stride=1)
            self.conv_rgb6 = ResBlock(channels_in=r_filter[4], num_filters=r_filter[5], stride=2)
            self.conv_rgb6_1 = ResBlock(channels_in=r_filter[5], num_filters=r_filter[5], stride=1)

            self.predict_mask = nn.Sequential(
                nn.Conv2d(r_filter[1]+d_filter[0]+1, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            )

        elif mode == 'N':
            s_filter = [32, r_filter[1]+d_filter[0]+1, r_filter[2]+d_filter[1]+1, r_filter[3]+d_filter[2]+1, r_filter[4]+d_filter[3]+1, 256]

            self.upsample4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            self.upsample3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            self.upsample2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            self.upsample1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)   

            self.predict_normal5 = nn.Conv2d(s_filter[5], 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal4 = nn.Conv2d(s_filter[4], 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal3 = nn.Conv2d(s_filter[3], 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal2 = nn.Conv2d(s_filter[2], 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal1 = nn.Conv2d(s_filter[1], 2, kernel_size=1, stride=1, padding=0, bias=True)

            self.conv_sparse1 = ResBlock(channels_in=2, num_filters=s_filter[0], stride=1)
            self.conv_sparse2 = ResBlock(channels_in=s_filter[0], num_filters=s_filter[1], stride=1)
            self.conv_sparse3 = ResBlock(channels_in=s_filter[1], num_filters=s_filter[2], stride=2)
            self.conv_sparse4 = ResBlock(channels_in=s_filter[2], num_filters=s_filter[3], stride=2)
            self.conv_sparse5 = ResBlock(channels_in=s_filter[3], num_filters=s_filter[4], stride=2)
            self.conv_sparse6 = ResBlock(channels_in=s_filter[4], num_filters=s_filter[5], stride=2)
        
            self.conv_rgb1 = ResBlock(channels_in=3, num_filters=r_filter[0], stride=1)
            self.conv_rgb2 = ResBlock(channels_in=r_filter[0], num_filters=r_filter[1], stride=1)
            self.conv_rgb3 = ResBlock(channels_in=r_filter[1], num_filters=r_filter[2], stride=2)
            self.conv_rgb3_1 = ResBlock(channels_in=r_filter[2], num_filters=r_filter[2], stride=1)
            self.conv_rgb4 = ResBlock(channels_in=r_filter[2], num_filters=r_filter[3], stride=2)
            self.conv_rgb4_1 = ResBlock(channels_in=r_filter[3], num_filters=r_filter[3], stride=1)
            self.conv_rgb5 = ResBlock(channels_in=r_filter[3], num_filters=r_filter[4], stride=2)
            self.conv_rgb5_1 = ResBlock(channels_in=r_filter[4], num_filters=r_filter[4], stride=1)
            self.conv_rgb6 = ResBlock(channels_in=r_filter[4], num_filters=r_filter[5], stride=2)
            self.conv_rgb6_1 = ResBlock(channels_in=r_filter[5], num_filters=r_filter[5], stride=1)

        #self.deconv4 = UpProject(r_filter[5], 256)
        #self.deconv3 = UpProject(s_filter[4], 128)
        #self.deconv2 = UpProject(s_filter[3], 64)
        #self.deconv1 = UpProject(s_filter[2], 32)
        self.deconv4 = nn.ConvTranspose2d(s_filter[5], d_filter[3], 4, 2, 1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(s_filter[4], d_filter[2], 4, 2, 1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(s_filter[3], d_filter[1], 4, 2, 1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(s_filter[2], d_filter[0], 4, 2, 1, bias=False)


    def forward(self, rgb, lidar, mask):
        b, c, w, h = rgb.size()

        sparse_input = torch.cat((lidar, mask), 1)
        s1 = self.conv_sparse1(sparse_input) # 256 x 512
        s2 = self.conv_sparse2(s1) # 128 x 256
        s3 = self.conv_sparse3(s2) # 64 x 128
        s4 = self.conv_sparse4(s3) # 32 x 64
        s5 = self.conv_sparse5(s4) # 16 x 32
        s6 = self.conv_sparse6(s5) # 8 x 16

        r1 = self.conv_rgb1(rgb) # 256 x 512
        r2 = self.conv_rgb2(r1) # 128 x 256
        r3 = self.conv_rgb3_1(self.conv_rgb3(r2)) # 64 x 128
        r4 = self.conv_rgb4_1(self.conv_rgb4(r3)) # 32 x 64
        r5 = self.conv_rgb5_1(self.conv_rgb5(r4)) # 16 x 32
        r6 = self.conv_rgb6_1(self.conv_rgb6(r5)) + s6  # 8 x 16

        o5 = self.upsample4(self.predict_normal5(r6)) # 16, 32
        cat5 = adaptive_cat(r5, self.deconv4(r6), o5) + s5 # 16, 32

        o4 = self.upsample3(self.predict_normal4(cat5))
        cat4 = adaptive_cat(r4, self.deconv3(cat5), o4) + s4 

        o3 = self.upsample2(self.predict_normal3(cat4))
        cat3 = adaptive_cat(r3, self.deconv2(cat4), o3) + s3 

        o2 = self.upsample1(self.predict_normal2(cat3))
        cat2 = adaptive_cat(r2, self.deconv1(cat3), o2) + s2 
        dense = self.predict_normal1(cat2)

        if self.mode == 'I':
            dense = F.interpolate(dense, (w, h), mode='bilinear', align_corners=True)
            return dense
        
        if self.mode == 'C':
            color_path_mask = self.predict_mask(cat2)
            return dense, color_path_mask, cat2

        if self.mode == 'N':
            return dense, cat2

class maskBlock(nn.Module):
    def __init__(self):
        super(maskBlock, self).__init__()
        self.mask_block = self.make_layers()

    def make_layers(self):
        in_channels = 81
        cfg = [81, 81]

        out_channels = 1
        layers = []

        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v

        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mask_block(x)

class deepLidar(nn.Module):
    def __init__(self):
        super(deepLidar, self).__init__()
        self.normal = deepCompletionUnit(mode='I')
        self.color_path = deepCompletionUnit(mode='C')
        self.normal_path = deepCompletionUnit(mode='N')
        self.mask_block_C = maskBlock()
        self.mask_block_N = maskBlock()

    def forward(self, rgb, lidar, mask, stage):
        surface_normal = self.normal(rgb, lidar, mask)
        if stage == 'N':
            return None, None, None, None, surface_normal

        color_path_dense, confident_mask, cat2C = self.color_path(rgb, lidar, mask)
        normal_path_dense, cat2N = self.normal_path(surface_normal, lidar, confident_mask)

        color_attn = self.mask_block_C(cat2C)
        normal_attn = self.mask_block_N(cat2N)

        return color_path_dense, normal_path_dense, color_attn, normal_attn, surface_normal

