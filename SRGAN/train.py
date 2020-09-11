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

        self.epochs = 25
        self.batch_size = 32
        self.gen_optimizer = Adam(gen.parameters(), lr = 1e-4, betas=(0.9, 0.999))
        self.gen_optimizer2 = Adam(gen.parameters(), lr = 1e-5, betas=(0.9, 0.999))
        self.dis_optimizer = Adam(dis.parameters(), lr = 1e-4, betas=(0.9, 0.999))

        self.contentloss = nn.MSELoss()

        self.dataloader = data 
        self.z = fixed_lr_images

        self.best_loss= 'na'


    def train_gen(self, hr,lr, conloss = 'mse'):

        ## Generate images and class output
        gen_hr = self.generator(lr)

        
        ## Content Loss
        if conloss == 'mse':
            con_loss = self.contentloss(gen_hr,hr)
        else:
            ## VGG Predictions
            vgg_hr  = self.vgg_model(hr) 
            vgg_lr = self.vgg_model(gen_hr) 

            con_loss = 1e-6  * self.contentloss(vgg_lr, vgg_hr) + self.contentloss(gen_hr,hr)

        ## Fake images for discriminator
        dis_out = self.discriminator(gen_hr)

        ## Adversarial Loss
        adv_loss = 1 - dis_out.mean()

        ##Overall Generator Loss 
        loss = con_loss + 1e-3 * adv_loss   

        ## Update the weights
        self.generator.zero_grad()
        loss.backward()
        if conloss == 'mse':
            self.gen_optimizer.step()
        else:
            self.gen_optimizer2.step()
            
        return loss 

    def train_disc(self, hr,lr):

        ## Generated Data
        gen_hr = self.generator(lr)

        ## Predictions from discriminator for real and fake chrominance channels
        real_dis_out = self.discriminator(hr)
        fake_dis_out = self.discriminator(gen_hr.detach())

        ## Compute Loss
        loss = 1 - real_dis_out.mean() + fake_dis_out.mean()
        
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
            cv2.imwrite('Image_Epoch:{}_{}.png'.format(epoch_no,i+1), np.transpose(img[i] * 255, (1,2,0)))

    def save_checkpoint(self, loss_value, epoch, mode = 'every_epoch'):

      if mode == 'best':
        if self.best_loss == 'na' :
            self.best_loss = loss_value
            torch.save(self.generator.state_dict(), '/content/Epoch_{}_Generator.pth'.format(epoch+1))
            print('Model Saved [Epoch:{}]'.format(epoch+1))
        else:
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                torch.save(self.generator.state_dict(), '/content/Epoch_{}_Generator.pth'.format(epoch+1))
                print('Model Saved [Epoch:{}]'.format(epoch+1))
      else:
        torch.save(self.generator.state_dict(), '/content/Epoch_{}_Generator.pth'.format(epoch+1))
        print('Model Saved [Epoch:{}]'.format(epoch+1))

    
    def train(self):

        self.gen_loss = []
        self.dis_loss = []
        
        ## Generator initialization
        for epoch in range(5):

            epoch_loss = {'gen_loss':[], 'dis_loss':[]}
            print('')
            print('[Epoch: {} / {}]'.format(epoch+1, self.epochs))
            
            eg_loss = []
            ed_loss = []

            for batch in tqdm(self.dataloader):

                hr,lr = batch['hr'].to(self.device)  , batch['lr'].to(self.device) 
                
                ## Retrive the Batch Size
                bs = hr.size(0)    

                ## Update the discrminator
                dis_loss = self.train_disc(hr, lr)
                ed_loss.append(dis_loss)            

                ## Update the generator
                gen_loss = self.train_gen(hr, lr)
                eg_loss.append(gen_loss)                

            self.gen_loss.append(torch.mean(torch.FloatTensor(eg_loss)))
            self.dis_loss.append(torch.mean(torch.FloatTensor(ed_loss)))

            ## Print Epoch Information
            print('[ Generator Loss: {} | Discriminator Loss: {} ] '.format(gen_loss, dis_loss))

            self.save_checkpoint(gen_loss, epoch)

            ## Plot predicted images to visualize images
            self.plot_images(self.z,epoch + 1)

        print('')
        
        print('Shifting to VGG Loss based training.....')

        ## VGG Loss Training
        for epoch in range(5,self.epochs):

            epoch_loss = {'gen_loss':[], 'dis_loss':[]}
            print('')
            print('[Epoch: {} / {}]'.format(epoch+1, self.epochs))
            
            eg_loss = []
            ed_loss = []
            
            for batch in tqdm(self.dataloader):

                hr,lr = batch['hr'].to(self.device)  , batch['lr'].to(self.device) 
                
                ## Retrive the Batch Size
                bs = hr.size(0)    

                ## Update the discrminator
                dis_loss = self.train_disc(hr, lr)
                ed_loss.append(dis_loss)            

                ## Update the generator
                gen_loss = self.train_gen(hr, lr, conloss='vgg')
                eg_loss.append(gen_loss)                

            self.gen_loss.append(torch.mean(torch.FloatTensor(eg_loss)))
            self.dis_loss.append(torch.mean(torch.FloatTensor(ed_loss)))

            ## Print Epoch Information
            print('[ Generator Loss: {} | Discriminator Loss: {} ] '.format(gen_loss, dis_loss))

            self.save_checkpoint(gen_loss, epoch)

            ## Plot predicted images to visualize images
            self.plot_images(self.z,epoch + 1)
