import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import sys
import glob
import cv2
import torch.nn as nn

from RFDN import RFDN # assuming that you have the RFDN model implementation available
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import utils_logger
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
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
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
    parser.add_argument('--model_path', type=str, default="/checkpoint2/model.pt")
    args = parser.parse_args(argv)
    return args

# define the PSNR function
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
def validation(argv):
    args = parse_args(argv)

    utils_logger.logger_info('validation_logs', log_path='validation log.log')
    logger = logging.getLogger('validation_log')

    
    # set the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the pretrained RFDN model
    model = RFDN()
    ckt = torch.load("checkpoint2/model.pt")
    model.load_state_dict(ckt, strict=False)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # set the model to evaluation mode
  

    # define the validation dataset and dataloader
    patch_size = tuple(map(int, args.patch_size.split('x')))
    dataset_valid = YUVDIV2KDataset(args.data_glob_valid, hr_patch_size=patch_size)
    # define the transform to preprocess the validation images

    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

 
    valid_loss = 0
    criterion = nn.MSELoss()
            
    # loop over the validation data and calculate the PSNR values
    with torch.no_grad():
        psnr_y_total = 0
        psnr_u_total = 0
        psnr_v_total = 0
        count = 0

        for data in dataloader_valid:
            # get the input and target images
            input_img = data[0].to(device)
            target_img = data[1].to(device)

            # generate the super-resolved image using the model
            output_img_yuv = model(input_img)
            loss = criterion(output_img_yuv, target_img)
            valid_loss += loss.item()
            output_img_y = output_img_yuv[:, 0, :, :]
            output_img_u = output_img_yuv[:, 1, :, :]
            output_img_v = output_img_yuv[:, 2, :, :]

            # calculate the PSNR for the Y, U, and V channels
            psnr_y = calculate_psnr(output_img_y.cpu().numpy().squeeze(), target_img[:, 0, :, :].cpu().numpy().squeeze())
            psnr_u = calculate_psnr(output_img_u.cpu().numpy().squeeze(), target_img[:, 1, :, :].cpu().numpy().squeeze())
            psnr_v = calculate_psnr(output_img_v.cpu().numpy().squeeze(), target_img[:, 2, :, :].cpu().numpy().squeeze())
            logger.info("psnr y: {} , psnr u: {} , psnr v: {}".format(psnr_y,psnr_u,psnr_v))

            # add the PSNR values to the total and update the count
            psnr_y_total += psnr_y
            psnr_u_total += psnr_u
            psnr_v_total += psnr_v
            count += 1

        valid_loss /= len(dataloader_valid.dataset)
        # calculate the average PSNR values
        avg_psnr_y = psnr_y_total / count
        avg_psnr_u = psnr_u_total / count
        avg_psnr_v = psnr_v_total / count
        logger.info(" valid_loss: {}, avg psnr y: {} , avg psnr u: {} , avg psnr v: {}".format(valid_loss,avg_psnr_y,avg_psnr_u,avg_psnr_v))
        # print the results
        print("Average PSNR (Y): {:.2f} dB".format(avg_psnr_y))
        print("Average PSNR (U): {:.2f} dB".format(avg_psnr_u))
        print("Average PSNR (V): {:.2f} dB".format(avg_psnr_v))
        
if __name__ == '__main__':
    validation(sys.argv[1:])