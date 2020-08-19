import torch 
from PIL import Image
from tqdm import tqdm
from torch.utils.data import  Dataset
from torchvision import datasets
import os 
import numpy as np

class  Dataprep(Dataset):

    def __init__(self, root, data = 'voc2012'):
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
                self.data_list.append(self.root + '/' + image)          

        ## Downscaling Factor
        self.ds_factor = 4
        

    def __len__(self):

        if self.tr_data == 'stl-10' or self.tr_data == 'imagenet':
            return len(self.data)
        else: 
            return len(self.data_list)
    
    def __getitem__(self,idx):
        
        if self.tr_data == 'stl-10' or self.tr_data == 'imagenet':
            img, _ = self.data[idx]
        else:
            img = Image.open(self.data_list[idx])

        ## Resize the image (HR)
        img = img.resize((224,224))
        h,w = img.size 
        
        img_n = img
        
        ## Scale the image (HR)
        img = np.asarray(img) / 255

        img = np.transpose(img, (2,0,1))        
        img = torch.tensor(np.asarray(img), dtype=torch.float32)

        ## Resize the image (LR)
        img_n = img_n.resize((int(h/self.ds_factor),int(w/self.ds_factor)))

        ## Scale the image (LR)
        img_n = np.asarray(img_n) /255
            
        img_n = np.transpose(img_n, (2,0,1))
        img_n = torch.tensor(img_n, dtype=torch.float32)

        return {
            'hr': img,
            'lr': img_n
        }