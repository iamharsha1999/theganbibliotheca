import torch 
import torch.nn as nn
import pretrainedmodels

class generator(nn.Module):

    def __init__(self):
        super().__init__()

        # Input Dimensions 224 x 224 x 3 (Grayscale Image)
        
        ## Features from VGG 16 
        
        self.vgg_model = pretrainedmodels.__dict__['vgg16'](pretrained = 'imagenet')
        self.vgg_model = nn.Sequential(*[self.vgg_model._features[i] for i in range(23)])
        # Output Dimensions 28 x 28 x 512

        ## Global Features
        self.global_features = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride = 2, padding = 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,512, kernel_size=3, stride = 1, padding = 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,512, kernel_size=3, stride = 2, padding = 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,512, kernel_size=3, stride = 1, padding = 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ) 
    
        
        self.global_features2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(51200, 1024),
            nn.Linear(1024,512),
            nn.Linear(512,256)
        )
        # After repetition and reshaping -> Output Dimension 28 x 28 x 256

        ## VGG-16 Imagenet Class
        self.global_featuresclass = nn.Sequential(
            nn.Flatten(),
            nn.Linear(51200,4096),
            nn.Linear(4096,4096),
            nn.Linear(4096,1000),
            nn.Softmax()
        )

        ## MidLevel Features
        self.midlevel_features  = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3, stride=1, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,256,kernel_size=3, stride=1, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(256)
        )

        ##Output Model
        self.output_model = nn.Sequential(
            nn.Conv2d(512,256, kernel_size=1, stride=1, padding =0, bias = False),
            nn.ReLU(inplace = False),

            nn.Conv2d(256,128,kernel_size=3, stride =1, padding =2, bias=False),
            nn.ReLU(inplace = True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128,64, kernel_size=3, stride=1, padding = 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding = 2),
            nn.ReLU(inplace=True),
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64,32, kernel_size=3, stride=1, padding = 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(32,2, kernel_size=3, stride = 1, padding = 2),
            nn.ReLU(inplace = True),

            nn.UpsamplingNearest2d(scale_factor=2)
        )
        
    def forward(self,x):

        ## x is a grayscale image -> Convert to 3 channel image
        x = x.repeat(1,3,1,1)

        x_features = self.vgg_model(x) 

        x_global_features = self.global_features(x_features)       
        
        x_global_features2 = self.global_features2(x_global_features)
        x_global_features2 =  x_global_features2.unsqueeze(2).repeat(1,1,28*28).view(-1,256,28,28)

        x_global_class = self.global_featuresclass(x_global_features)

        x_midlevel_features = self.midlevel_features(x_features)

        x_fused_features = torch.cat((x_global_features2, x_midlevel_features ), dim = 1)

        x_output = self.output_model(x_fused_features)

        return x_global_class, x_output
      

class discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=4, padding = 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, padding = 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, padding = 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, padding = 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, padding =1, bias=False)
            )

    def forward(self,x,y):

        x = torch.cat(x,y)

        x = self.model(x)

        return x 



        
