import torch 
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
import cv2
import pretrainedmodels 
import matplotlib.pyplot as plt 
import numpy as np 

class Trainer():

    def __init__(self, gen , dis,data, fixed_lr_images,device = 'cuda'):

        self.device = device 
        self.generator = gen 
        self.discriminator = dis
        self.vgg_model = pretrainedmodels.__dict__['vgg16'](pretrained = 'imagenet')._features.to(device)


        ## Device Initialization
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Device Name:', torch.cuda.get_device_name(0))
        else:
            self.device = torch.device('cpu')
        
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.epochs = 10
        self.batch_size = 32
        self.gen_optimizer = Adam(gen.parameters(), lr = 2e-5, betas=(0.5, 0.999))
        self.dis_optimizer = Adam(dis.parameters(), lr = 2e-5, betas=(0.5, 0.999))

        self.contentloss = nn.MSELoss() 
        self.advloss = nn.BCEWithLogitsLoss(reduction='sum')

        self.lambda_adv = 1e-3
        self.dataloader = data 
        self.z = fixed_lr_images

    def train_gen(self, hr,lr):

        ## Generate images and class output
        gen_hr = self.generator(lr)

        ## Content Loss
        vgg_hr = self.vgg_model(hr)
        vgg_lr  = self.vgg_model(gen_hr)
        con_loss = self.contentloss(vgg_lr, vgg_hr)

        ## Fake images for discriminator
        dis_out = self.discriminator(gen_hr)

        ## Adversarial Loss
        adv_loss = self.advloss(dis_out, torch.ones_like(dis_out).to(self.device))

        ##Overall Generator Loss 
        loss = con_loss + self.lambda_adv* adv_loss   

        ## Update the weights
        self.generator.zero_grad()
        loss.backward()
        self.gen_optimizer.step()

        return loss 

    def train_disc(self, hr,lr):

        ## Generated Data
        gen_hr = self.generator(lr)

        ## Predictions from discriminator for real and fake chrominance channels
        real_dis_out = self.discriminator(hr)
        fake_dis_out = self.discriminator(gen_hr.detach())

        ## Compute Loss
        real_loss = self.contentloss(real_dis_out, torch.ones_like(real_dis_out).to(self.device))
        fake_loss = self.contentloss(fake_dis_out, torch.zeros_like(fake_dis_out).to(self.device))
        loss = real_loss + fake_loss
        
        ##Update the weights
        self.discriminator.zero_grad()
        loss.backward()
        self.dis_optimizer.step()

        return loss

    def plot_images(self,lr, epoch_no):
        lr = torch.tensor(lr, dtype = torch.float32).to(self.device)
        with torch.no_grad():
            gen_hr = self.generator(lr)
        img = gen_hr.to('cpu').numpy()
        for i in range(len(img)):        
            plt.imshow(np.transpose(img[i], (1,2,0)), interpolation = 'none')
            plt.savefig('Image_Epoch:{}_{}.png'.format(epoch_no+1,i+1))
        
    
    def train(self):

        self.gen_loss = []
        self.dis_loss = []

        for epoch in range(self.epochs):

            epoch_loss = {'gen_loss':[], 'dis_loss':[]}
            print('[Epoch: {} / {}]'.format(epoch+1, self.epochs))
            
            eg_loss = []
            ed_loss = []
            
            for batch in tqdm(self.dataloader):

                hr,lr = batch['hr'].to(self.device),batch['lr'].to(self.device)
                
                ## Retrive the Batch Size
                bs = hr.size(0)                

                ## Update the generator
                gen_loss = self.train_gen(hr, lr)
                eg_loss.append(gen_loss)

                ## Update the discrminator
                dis_loss = self.train_disc(hr, lr)
                ed_loss.append(dis_loss)

            self.gen_loss.append(torch.mean(torch.FloatTensor(eg_loss)))
            self.dis_loss.append(torch.mean(torch.FloatTensor(ed_loss)))

            ## Print Epoch Information
            print('[ Generator Loss: {} | Discriminator Loss: {} ] '.format(gen_loss, dis_loss))

            ## Plot predicted images to visualize images
            self.plot_images(self.z,epoch + 1)
