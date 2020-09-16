import torch 
import cv2
from tqdm import tqdm
from torch.utils.data import  Dataset
from torchvision import datasets
import os 
import numpy as np
from albumentations import  Resize, Normalize, Compose
from albumentations.pytorch import ToTensor

class  Dataprep(Dataset):

    def __init__(self, root, img_size = 64, data = 'celeb-a'):
        super().__init__()
       
        
        self.root = root 
        self.img_size = img_size

        ## Train Set
        self.tr_data = data 

        ## Transform functions
        self.transform = Compose([Resize( self.img_size,  self.img_size),
                                                            Normalize(mean = [0.5], std = [0.5]
                                                            )])

        if self.tr_data == 'celeb-a':
            self.data = datasets.CelebA(self.root, split = 'train',   download = True)
        elif self.tr_data == 'mnist':
            self.data = datasets.MNIST(self.root, train = True,  download =  True)
        else:            
            self.data_list = []
            for image in os.listdir(self.root):
                self.data_list.append(self.root + '/' + image)          

        
    
    def __len__(self):

        if self.tr_data == 'celeb-a' or self.tr_data == 'mnist':
            return len(self.data)
        else:
            return  len(self.data_list)
    
    def __getitem__(self, idx):

        if self.tr_data == 'celeb-a':
            img = self.transform(image = np.array(self.data[idx][0]))['image']
            img = torch.tensor(img, dtype = torch.float32).view(-1,self.img_size, self.img_size)
            return img
        elif self.tr_data == 'mnist':
            img = self.transform(image = np.array(self.data[idx][0]))['image']
            img = torch.tensor(img, dtype = torch.float32).view(-1,self.img_size, self.img_size)
            return img
        else:
            img = cv2.imread(self.data_list[idx])            
            img = self.transform(image = img)['image']
            img = np.transpose(img, (2,0,1))
            img = torch.tensor(img, dtype = torch.float32).view(-1,self.img_size, self.img_size)
            return img 

       

