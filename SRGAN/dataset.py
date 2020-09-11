import torch 
import cv2
from tqdm import tqdm
from torch.utils.data import  Dataset
from torchvision import datasets
import os 
import numpy as np
from albumentations import Compose, RandomCrop, Resize


class  Dataprep(Dataset):

    def __init__(self, root, crop_size = 96, data = 'voc2012'):
        super().__init__()
       
        ## Root Directory 
        self.root = root 

        ## Train Set
        self.tr_data = data 

        if self.tr_data == 'stl-10':
            self.data = datasets.STL10(self.root, split = 'train', download = True)
        elif self.tr_data == 'imagenet':
            self.data = datasets.ImageNet(self.root, split = 'train', download = True)
        else:            
            self.data_list = []
            for image in os.listdir(self.root):
                h,w,_  = cv2.imread(self.root + '/' + image).shape
                if h > 96 and w > 96:
                    self.data_list.append(self.root + '/' + image)          

        ## Downscaling Factor
        self.ds_factor = 4

        ## Transform functions
        self.hr_transform = Compose([ RandomCrop(crop_size, crop_size) ])

        self.lr_transform = Compose([ Resize( width = crop_size//4, height = crop_size//4 )])
        

    def __len__(self):

        if self.tr_data == 'stl-10' or self.tr_data == 'imagenet':
            return len(self.data)
        else: 
            return len(self.data_list)
    
    def __getitem__(self,idx):
        
        if self.tr_data == 'stl-10' or self.tr_data == 'imagenet':
            img, _ = self.data[idx]
        else:
            img = cv2.imread(self.data_list[idx])       
        
        img_hr = self.hr_transform(image = img)['image'] 
        img_lr =  self.lr_transform(image =  img_hr)['image'] 

        img_hr = img_hr / 255
        img_lr = img_lr / 255
        
        img_hr = np.transpose(img_hr, (2,0,1))
        img_lr = np.transpose(img_lr, (2,0,1))        
        img_hr = torch.tensor(img_hr, dtype=torch.float32)
        img_lr = torch.tensor(img_lr, dtype=torch.float32)        
       
        return {
            'hr': img_hr,
            'lr':  img_lr
        }