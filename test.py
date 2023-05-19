
import numpy as np
import torch
import torch.nn as nn
import argparse
import logging
import utils_logger
import cv2
import os
import subprocess as sp
from RFDN import RFDN
from PIL import Image
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
import time

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--test_data_lr', type=str, default='test_data/LR_YUV420/Bosphorus_1920x1080_120F.yuv')
    parser.add_argument('--test_data_hr', type=str, default='test_data/HR_YUV420/Bosphorus_3840x2160_120F.yuv')
    parser.add_argument('--patch_size', type=str, default='256x256')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--model_path', type=str, default="checkpoint2/model.pt")
    parser.add_argument('--interpolate', type=str, default="bilinear")
    args = parser.parse_args(argv)
    return args



def calculate_psnr(y_true, y_pred):
    """
    Calculate PSNR for Y, U, V channels in YUV444 format
    """
    y_true_y, u_true, v_true = torch.chunk(y_true, chunks=3, dim=1)
    y_pred_y, u_pred, v_pred = torch.chunk(y_pred, chunks=3, dim=1)

    # Calculate MSE and PSNR for Y, U, and V components
    mse_y = torch.mean((y_true_y - y_pred_y)**2)
    mse_u = torch.mean((u_true - u_pred)**2)
    mse_v = torch.mean((v_true - v_pred)**2)

    psnr_y = 10 * torch.log10(1 / mse_y)
    psnr_u = 10 * torch.log10(1 / mse_u)
    psnr_v = 10 * torch.log10(1 / mse_v)
    return psnr_y.item(), psnr_u.item(), psnr_v.item()


 
def main(argv):
    args = parse_args(argv)
    log_name = ""
    if args.interpolate == "bilinear":
        log_name+="testing_bilinear_logs"
    else:
        log_name+="testing_bicubic_logs"
    utils_logger.logger_info(log_name, log_path=log_name+'.log')
    logger = logging.getLogger(log_name)

    

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device ='cpu'
    #Loading the model
    model = RFDN()
    ckt = torch.load(args.model_path)
    model.load_state_dict(ckt, strict=False)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    width, height =1920,1080
    width_hr, height_hr = 3840,2160
    # Number of frames: in YUV420 frame size in bytes is width*height*1.5
    n_frames = 120
    print("frames:",n_frames)
    # Open 'input.yuv' a binary file.
    f_lr = open(args.test_data_lr, 'rb')
    f_hr =open(args.test_data_hr,'rb')

    
    frame_size_lr = int(width * height * 1.5)
    frame_size_hr = int(width_hr * height_hr * 1.5)
    i=0
    SR = []
    PSNR_Y = []
    PSNR_U = []
    PSNR_V = []
    tot_time_start = time.time()
    while True:

        # Read a single YUV420 frame
        yuv420_frame_lr = f_lr.read(frame_size_lr)
        yuv420_frame_hr = f_hr.read(frame_size_hr)
        # If we've reached the end of the video file, break out of the loop
        if not( yuv420_frame_lr and yuv420_frame_hr):
            break
        # Reshape the YUV420 frame into a 3D numpy array
        yuv420_frame_lr = np.frombuffer(yuv420_frame_lr, dtype=np.uint8).reshape((int(height * 1.5), width))
        rgb_lr = cv2.cvtColor(yuv420_frame_lr, cv2.COLOR_YUV2RGB_I420)
        cv2.imwrite('output/LR/'+str(i)+'.png',rgb_lr)

        yuv420_frame_hr = np.frombuffer(yuv420_frame_hr, dtype=np.uint8).reshape((int(height_hr * 1.5), width_hr))
        bgr = cv2.cvtColor(yuv420_frame_hr, cv2.COLOR_YUV2BGR_I420)

        cv2.imwrite('output/HR/'+str(i)+'.png',bgr)
        #print(yuv420_frame_lr.shape)
        # Extract the Y, U, and V components from the YUV420 frame
        yuv_420_444_s = time.time()
        y =( yuv420_frame_lr[:height, :]).reshape((height, width))
        u = ( yuv420_frame_lr[height:int(height*1.25), :]).reshape((height//2, width//2))
        v = (yuv420_frame_lr[int(height*1.25):, :]).reshape((height//2, width//2))

    
        # Upsample the U and V components to YUV444 (bilinear) for bicubic INTER_CUBIC
        if args.interpolate =="bilinear":
            u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
        elif args.interpolate =="bicubic":
            u = cv2.resize(u, (width, height), interpolation=cv2.INTER_CUBIC)
            v = cv2.resize(v, (width, height), interpolation=cv2.INTER_CUBIC)
        

        yuv = np.dstack((y,u,v))

        #print(yuv.shape)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite('test.png',bgr)
        y= torch.from_numpy(y).unsqueeze(0).float()
        u=torch.from_numpy(u).unsqueeze(0).float()
        v =torch.from_numpy(v).unsqueeze(0).float()

       
        #Combine the Y, U, and V components into a YUV444 frame
        yuv444 = torch.cat([y,u,v]).unsqueeze(0).permute(0, 1, 2,3).float().to(device)
        yuv_420_444_e = time.time() -yuv_420_444_s
        print("yuv 420 to yuv 444", yuv_420_444_e)
        yuv444=yuv444/255.
        #yuv = torch.from_numpy(yuv).unsqueeze(0).permute(0,3,1,2).float().to(device)
        #print(yuv.shape)
        yuv_model_s = time.time()
        output_444 = model(yuv444)
        yuv_model_e =time.time() -yuv_model_s
        print("model runtime:", yuv_model_e)
        output_444 = output_444.squeeze(0)

        # Copy the chroma values to the U and V channels of the YUV 420 planar image
        yuv_444_420_s = time.time()
        
        output_444 = output_444.cpu().numpy()
        output = output_444.transpose((1, 2, 0)) 
        y= output[:,:,0]
        u= output[:,:,1]
        v=output[:,:,2]
        bgr = cv2.cvtColor(output, cv2.COLOR_YUV2BGR)*255  
        #rgb = cv2.cvtColor(output_444, cv2.COLOR_YUV2RGB_I420)
        cv2.imwrite('output/SR/'+str(i)+'.png',bgr)
        i+=1

        #converting y,u and v from yuv444 to yuv420
        #downsample u and v
        y_sr = y
        u_sr = cv2.resize(u, (width_hr//2, height_hr//2), interpolation=cv2.INTER_AREA)
        v_sr = cv2.resize(v, (width_hr//2, height_hr//2), interpolation=cv2.INTER_AREA) 
        yuv_444_420_e = time.time() - yuv_444_420_s
        print("yuv 420 to yuv 444", yuv_444_420_e)
        y_hr =( yuv420_frame_hr[:height_hr, :]).reshape((height_hr, width_hr))/255
        u_hr = ( yuv420_frame_hr[height_hr:int(height_hr*1.25), :]).reshape((height_hr//2, width_hr//2))/255
        v_hr = (yuv420_frame_hr[int(height_hr*1.25):, :]).reshape((height_hr//2, width_hr//2))/255

        mse_y = np.mean((y_hr - y_sr) ** 2)
        mse_u = np.mean((u_hr - u_sr) ** 2)
        mse_v = np.mean((v_hr - v_sr) ** 2)

        psnr_y=  20 * np.log10(1/ np.sqrt(mse_y))
        psnr_u= 20 * np.log10(1/ np.sqrt(mse_u))
        psnr_v= 20 * np.log10(1/ np.sqrt(mse_v))
        

        logger.info("psnr y: {} , psnr u: {} , psnr v: {}".format(psnr_y,psnr_u,psnr_v))
        PSNR_Y.append(psnr_y)
        PSNR_U.append(psnr_u)
        PSNR_V.append(psnr_v)
    tot_time_end = time.time()-tot_time_start
    print("total testing time", tot_time_end)
    f_lr.close()
    f_hr.close()
    avg_psnr_y = sum(PSNR_Y)/len(PSNR_Y)
    avg_psnr_u = sum(PSNR_U)/len(PSNR_U)
    avg_psnr_v = sum(PSNR_V)/len(PSNR_V)
    print(" Average of psnr y: {} , psnr u: {} , psnr v: {}".format(avg_psnr_y,avg_psnr_u,avg_psnr_v))

if __name__ == '__main__':
    main(sys.argv[1:])