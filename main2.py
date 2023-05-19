import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import RFDN as m
#from network import InformedDenoiseModel
import os
import cv2
import sys
import glob
import matplotlib.pyplot as plt
import logging
import utils_logger
import time
import argparse
class YUVDIV2KDataset(Dataset):
    def __init__(self, img_glob, hr_patch_size=(512, 512)):
        self.imgs = sorted(glob.glob(img_glob))
        self.hr_patch_size = hr_patch_size
        self._yuv_from_rgb_matrix = np.array([
            [0.299, -0.14714119, 0.61497538],
            [0.587, -0.28886916, -0.51496512],
            [0.114, 0.43601035, -0.10001026]
        ], dtype=np.float32)
        self._yuv_from_rgb_offset = np.array([0, 0.5, 0.5], dtype=np.float32)
        self.lr_patch_size = (hr_patch_size[0] // 2, hr_patch_size[1] // 2)
    
    def _get_y_uv_crop(self, y, uv, anchor, patch_size):
        y_crop = y[anchor[0]:anchor[0] + patch_size[0], anchor[1]:anchor[1] + patch_size[1]]
        y_crop = torch.from_numpy(y_crop).unsqueeze(-1)
        #uv_crop = uv[anchor[0] // 2:anchor[0] // 2 + patch_size[0] // 2, anchor[1] // 2:anchor[1] // 2 + patch_size[1] // 2]
        uv_crop = uv[anchor[0]:anchor[0] + patch_size[0], anchor[1]:anchor[1] + patch_size[1]]
        uv_crop = torch.from_numpy(uv_crop)
        return y_crop.permute(2, 0, 1), uv_crop.permute(2, 0, 1)
    
    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)/255.
        yuv = img @ self._yuv_from_rgb_matrix + self._yuv_from_rgb_offset
        y = yuv[:, :, 0]
        # uv = cv2.resize(
        #     yuv[:, :, 1:], (yuv.shape[1] // 2, yuv.shape[0] // 2)) 
        uv = yuv[:, :, 1:] 
        anchor_high_x = np.random.randint(
            0, y.shape[0] - self.hr_patch_size[0]+1)
        anchor_high_y = np.random.randint(
            0, y.shape[1] - self.hr_patch_size[1]+1)
        y_high, uv_high = self._get_y_uv_crop(
            y, uv, (anchor_high_x, anchor_high_y), self.hr_patch_size)
        y_low = F.interpolate(y_high.unsqueeze(0), size=self.lr_patch_size, mode='bilinear', align_corners=False).squeeze(0)
        uv_low = F.interpolate(uv_high.unsqueeze(0),
                               size=(self.lr_patch_size[0] , self.lr_patch_size[1]),
                               mode='bilinear', align_corners=False).squeeze(0)
        # uv_low = cv2.normalize(uv, None, -128, 127, cv2.NORM_MINMAX)  
        # y_low = cv2.normalize(uv, None, 0, 255, cv2.NORM_MINMAX) 
        # uv_high = cv2.normalize(uv, None, -128, 127, cv2.NORM_MINMAX)  
        # y_high = cv2.normalize(uv, None, 0, 255, cv2.NORM_MINMAX) 
        yuv_low = torch.cat([y_low, uv_low])
        yuv_high = torch.cat([y_high, uv_high])   
        return yuv_low, yuv_high
    
    def __len__(self) -> int:
        return len(self.imgs)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='checkpoint2')
    parser.add_argument('--data_glob_train', type=str, default='data/train/*.png')
    parser.add_argument('--data_glob_valid', type=str, default='data/valid/*.png')
    parser.add_argument('--patch_size', type=str, default='256x256')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--inner_dim', type=int, default=14)
    parser.add_argument('--lanczos', action='store_true', default=False)
    parser.add_argument('--lmbda', type=float, default=0.02)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--test_bicubic', type=str, default="False")
    parser.add_argument('--device', type=str, default="cuda:0")
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
    utils_logger.logger_info('training_logs2', log_path='training_logs2.log')
    logger = logging.getLogger('training_logs2')

    args = parse_args(argv)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("Loading data")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_size = tuple(map(int, args.patch_size.split('x')))
    dataset_train = YUVDIV2KDataset(args.data_glob_train, hr_patch_size=patch_size)
    dataset_valid = YUVDIV2KDataset(args.data_glob_valid, hr_patch_size=patch_size)
    

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.test_bicubic:
         psnr_y_tot = 0
         psnr_u_tot = 0
         psnr_v_tot = 0
         batch=0
         for batch_idx, (yuv_low, yuv_high) in enumerate(dataloader_train):
             batch+=1
             yuv_low = yuv_low.to(args.device)
             yuv_high = yuv_high.to(args.device)

             y= yuv_low[:,0,:,:].unsqueeze(0)
             u=yuv_low[:,1,:,:].unsqueeze(0)
             v=yuv_low[:,2,:,:].unsqueeze(0)
            
             y = F.interpolate(y, scale_factor= 2, mode='bicubic', align_corners=False)
             u=F.interpolate(u, scale_factor = 2, mode='bicubic', align_corners=False)
             v=F.interpolate(v, scale_factor = 2, mode='bicubic', align_corners=False)
             yuv = torch.cat([y,u,v]).permute(1,0,2,3)
             psnr_y, psnr_u, psnr_v = calculate_psnr(yuv_high, yuv)
             psnr_y_tot += psnr_y
             psnr_u_tot += psnr_u
             psnr_v_tot += psnr_v
             print("psnr y:",psnr_y," psnr u:",psnr_u, " psnr v:", psnr_v)
         psnr_y_tot /= batch 
         psnr_u_tot /= batch 
         psnr_v_tot /= batch 
         print("psnr y avg:",psnr_y_tot," psnr u avg:",psnr_u_tot, " psnr v avg:", psnr_v_tot)
         sys.exit(0)
             

    print("Running on",device)
    model = m.make_model(args)
    if args.model_path !="":
        model.load_state_dict(torch.load(args.model_path))
        
    model = model.to(device)

    # Define the loss function and optimizer
    learning_rate = args.lr
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = args.num_epochs
    print("lr:",args.lr," num_workers:",args.num_workers," batch size:", args.batch_size," epochs:", args.num_epochs)
    logger.info('lr:{} num_workers:{} batch_size:{} epochs:{} '.format(learning_rate,args.num_workers, args.batch_size,args.num_epochs))
    print("Training...")
    # Training loop
    train_losses = []
    valid_losses = []
    psnr_y=[]
    psnr_u = []
    psnr_v=[]

    for epoch in range(num_epochs):
    # Training   
        model.train()
        train_loss = 0

        for batch_idx, (yuv_low, yuv_high) in enumerate(dataloader_train):

            # Upsample the UV channel to match the size of the Y channel
            # uv_low_temp = F.interpolate(uv_low, scale_factor=2, mode='bilinear', align_corners=False)
            # uv_high_temp = F.interpolate(uv_high, scale_factor=2, mode='bilinear', align_corners=False
            yuv_low = yuv_low.to(device)
            yuv_high = yuv_high.to(device)
            optimizer.zero_grad()
            yuv_pred = model(yuv_low)
            loss = criterion(yuv_pred, yuv_high)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 5 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(yuv_low)}/{len(dataloader_train.dataset)} '
                        f'({100. * batch_idx / len(dataloader_train):.0f}%)]\tLoss: {loss.item():.6f}')
        train_loss = train_loss / len(dataloader_train.dataset)
        train_losses.append(train_loss)
        # Validation
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for yuv_low, yuv_high in dataloader_valid:
                yuv_low = yuv_low.to(device)
                yuv_high = yuv_high.to(device)
                yuv_pred = model(yuv_low)
                loss = criterion(yuv_pred, yuv_high)
                valid_loss += loss.item()
        
            
        valid_loss /= len(dataloader_valid.dataset)
        valid_losses.append(valid_loss)
        y,u,v =calculate_psnr(yuv_high,yuv_pred)
        psnr_y.append(y)
        psnr_u.append(u)
        psnr_v.append(v)
        print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Valid Loss = {valid_loss:.6f}')
        logger.info('epoch:{}, training loss:{}, validation loss:{}, psnr y:{}, psnr u:{}, psnr v:{},'.format(epoch,train_loss,valid_loss,y,u,v))
        if epoch % 5 == 0:
            #saving the modela
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model.pt'))
        if epoch%50==0:
            #plot and save loss graph
            plt.plot(train_losses, label='Training Loss')
            plt.plot(valid_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('checkpoint2/loss_graph.png')  # Save the Loss Graph Image
            plt.clf()
            #plot and save psnr graph
            
            plt.plot(psnr_y, label='y PSNR')
            plt.plot(psnr_u, label='u PSNR')
            plt.plot(psnr_v, label='v PSNR')
            plt.xlabel('Epoch')
            plt.ylabel('PSNR')
            plt.legend()
            plt.savefig('checkpoint2/validation_psnr_graph.png')
            plt.clf()
    
    

if __name__ == '__main__':
    main(sys.argv[1:])