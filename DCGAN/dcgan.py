import torch 
import  torch.nn as nn 


class Generator(nn.Module):
    def __init__(self, zdims,out_dims=1):
        super().__init__()

        self.gen = nn.Sequential(
                                                        ## Block 1
                                                        nn.ConvTranspose2d(zdims, 1024, kernel_size=4, stride = 1,padding =0, bias=False),
                                                        nn.BatchNorm2d(1024),
                                                        nn.ReLU(inplace=True),

                                                        ## Block 2
                                                        nn.ConvTranspose2d(1024, 512, kernel_size=8, stride = 2,padding =1, bias=False),
                                                        nn.BatchNorm2d(512),
                                                        nn.ReLU(inplace=True),

                                                        ## Block 3
                                                        nn.ConvTranspose2d(512, 256, kernel_size=16, stride = 2,padding =1, bias=False),
                                                        nn.BatchNorm2d(256),
                                                        nn.ReLU(inplace=True),

                                                        ## Block 4
                                                        nn.ConvTranspose2d(256, 128, kernel_size=32, stride = 2,padding =1, bias=False),
                                                        nn.BatchNorm2d(128),
                                                        nn.ReLU(inplace=True),
                                                        
                                                        ## Output Layer
                                                        nn.ConvTranspose2d(128, out_dims, kernel_size=64, stride = 2,padding =1, bias=False),
                                                        nn.Tanh()
        )
    
    def forward(self, x):

        return self.gen(x)

class Discriminator(nn.Module):

    def __init__(self, inp_dims):

        super().__init__()

        self.dis = nn.Sequential(
                                                        ## Block 1
                                                        nn.Conv2d(inp_dims,64, kernel_size = 4, stride =2, padding = 1, bias = False),
                                                        nn.LeakyReLU(0.2,inplace=True),

                                                        ## Block 2
                                                        nn.Conv2d(64,128, kernel_size = 4, stride =2, padding = 1, bias = False),
                                                        nn.BatchNorm2d(128),
                                                        nn.LeakyReLU(0.2,inplace=True),

                                                        ## Block 3
                                                        nn.Conv2d(128,256, kernel_size = 4, stride =2, padding = 1, bias = False),
                                                        nn.BatchNorm2d(256),
                                                        nn.LeakyReLU(0.2,inplace=True),

                                                        ## Block 4
                                                        nn.Conv2d(256,512, kernel_size = 4, stride =2, padding = 1, bias = False),
                                                        nn.BatchNorm2d(512),
                                                        nn.LeakyReLU(0.2,inplace=True),

                                                        ## Output Layer
                                                        nn.Conv2d(512,1, kernel_size = 4, stride =1, padding = 0, bias = False),
                                                        nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)

## Function to initializa weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
