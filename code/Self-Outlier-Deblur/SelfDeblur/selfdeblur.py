
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from SelfDeblur.networks.skip import skip
from SelfDeblur.networks.fcn import fcn
import PIL.Image as Image
import torch
import torch.optim
import glob
from skimage.io import imread
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from SelfDeblur.utils.common_utils import *
from SelfDeblur.utils.SSIM import SSIM

def SelfDeblur():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iter', type=int, default=600, help='number of epochs of training')
    parser.add_argument('--img_size', type=int, default=[800, 800], help='size of each image dimension')
    parser.add_argument('--kernel_size', type=int, default=[17, 17], help='size of blur kernel [height, width]')
    parser.add_argument('--data_path', type=str, default="datasets/", help='path to blurry image')
    parser.add_argument('--save_path', type=str, default="results/", help='path to save results')
    parser.add_argument('--save_frequency', type=int, default=20, help='lfrequency to save results')
    opt = parser.parse_args()
    #print(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark =True
    dtype = torch.cuda.FloatTensor

    warnings.filterwarnings("ignore")

    files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
    files_source.sort()
    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)


    # start #image
    for f in files_source:
        INPUT = 'noise'
        pad = 'reflection'
        LR = 0.01
        num_iter = opt.num_iter
        reg_noise_std = 0.001

        path_to_image = f
        imgname = os.path.basename(f)
        imgname = os.path.splitext(imgname)[0]

        #通过文件名判断模糊核大小
        if imgname.find('kernel1') != -1:
            opt.kernel_size = [17, 17]
        if imgname.find('kernel2') != -1:
            opt.kernel_size = [15, 15]
        if imgname.find('kernel3') != -1:
            opt.kernel_size = [13, 13]
        if imgname.find('kernel4') != -1:
            opt.kernel_size = [27, 27]
        if imgname.find('kernel5') != -1:
            opt.kernel_size = [11, 11]
        if imgname.find('kernel6') != -1:
            opt.kernel_size = [19, 19]
        if imgname.find('kernel7') != -1:
            opt.kernel_size = [21, 21]
        if imgname.find('kernel8') != -1:
            opt.kernel_size = [21, 21]

        _, imgs = get_image(path_to_image, -1) # load image and convert to np.
        y = np_to_torch(imgs).type(dtype)

        img_size = imgs.shape
        print(imgname)
        # ######################################################################
        padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
        opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw

        '''
        x_net:用于估计潜在图像
        '''
        input_depth = 4    #8

        net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

        net = skip( input_depth, 1,
                    num_channels_down = [64, 64, 64, 64, 64],
                    num_channels_up = [64, 64, 64, 64, 64],
                    num_channels_skip = [8, 8, 8, 8, 8],
                    upsample_mode='bilinear',
                    need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        net = net.type(dtype)

        '''
        k_net:用于估计模糊核
        '''
        n_k = 50    #200
        net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
        net_input_kernel.squeeze_()

        net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
        net_kernel = net_kernel.type(dtype)

        # Losses
        mse = torch.nn.MSELoss().type(dtype)
        ssim = SSIM().type(dtype)

        # optimizer
        optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
        scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

        # initilization inputs
        net_input_saved = net_input.detach().clone()
        net_input_kernel_saved = net_input_kernel.detach().clone()

        ### start SelfDeblur
        for step in tqdm(range(num_iter)):


            torch.cuda.empty_cache()
            # input regularization
            net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()

            # change the learning rate
            scheduler.step(step)
            optimizer.zero_grad()

            # get the network output
            out_x = net(net_input)
            out_k = net_kernel(net_input_kernel)
        
            out_k_m = out_k.view(-1,1,opt.kernel_size[0],opt.kernel_size[1])
            #中心化核
            if step > 400:
                out_k_m = adjust_kernel_center(out_k_m, opt.kernel_size[0], opt.kernel_size[1])
            # print(out_k_m)
            out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)
            #如果step小于1000，使用MSE损失，否则使用SSIM损失
            if step < 1000:
                total_loss = mse(out_y,y) 
            else:
                total_loss = 1-ssim(out_y, y) 

            total_loss.backward()
            optimizer.step()

            if (step+1) % opt.save_frequency == 0:
                #print('Iteration %05d' %(step+1))


                save_path = os.path.join(opt.save_path, 'latent.png')
                out_x_np = out_x.to('cpu').detach().numpy()
                out_x_np = out_x_np.squeeze()
                out_x_np = out_x_np[padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
                # 归一化
                out_x_np_normalized = (out_x_np - np.min(out_x_np)) / (np.max(out_x_np) - np.min(out_x_np))
                out_x_np_8 = (out_x_np_normalized * 255).astype(np.uint8)  # 转换为8位整数
                out_x_np_32 = (out_x_np_normalized * 255).astype(np.float32)
                #从NumPy数组创建PIL图像对象
                img = Image.fromarray(out_x_np_8)
                img.save(save_path)
                

                save_path = os.path.join(opt.save_path, 'kernel.png')
                out_k_np = out_k_m.to('cpu').detach().numpy()
                out_k_np = out_k_np.squeeze()
                out_k_np /= np.max(out_k_np)# 归一化
                out_k_np_8 = (out_k_np * 255).astype(np.uint8)
                out_k_np_32 = (out_k_np ).astype(np.float32)  # 转换为8位整数
                #从NumPy数组创建PIL图像对象
                img = Image.fromarray(out_k_np_8)
                img.save(save_path)

                # Save out_k as csv
                out_k_csv_path = os.path.join(opt.save_path, 'kernel.csv')
                np.savetxt(out_k_csv_path, out_k_np_32, delimiter=',')

    print("out_x_np.shape, out_k_m_np.shape: ", out_x_np_32.shape, out_k_np_32.shape)
    return out_x_np_32, out_k_np_32


if __name__ == '__main__':
    SelfDeblur()
