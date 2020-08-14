import torch 
import cv2
from tqdm import tqdm
from torch.utils.data import  Dataset
from torchvision import datasets
import os 

class  Dataprep(Dataset):

    def __init__(self, root, mean =0, var = 0.1, sigma =25, data = 'stl-10'):
        super().__init__()

        ## Root Directory 
        self.root = root 

        self.mean = 0
        self.var = 0.1
        self.sigma = 25

        if data == 'stl-10':
            self.data = datasets.STL10(self.root, split = 'train', download = True)
        
        ## Downscaling Factor
        self.ds_factor = 4

    def __len__(self):

        return len(self.data)
    
    def gauss_noise(self,image):
        row, col, ch = image.shape
        
        gauss = np.random.normal(self.mean,self.sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss

        return noisy

    def __getitem__(self,idx):

        img_n = self.data[idx][0].numpy()
        img_n = cv2.resize(img_n, (img_n.size()[0]/self.ds_factor, img_n.size()[1]/self.ds_factor))

        ## Gaussian Noise 
        img_n = self.gauss_noise(img_n)
        img = torch.tensor(img_n, dtype=torch.float32)
        
        return {'hr':img[idx],
                        'lr':img_n}

