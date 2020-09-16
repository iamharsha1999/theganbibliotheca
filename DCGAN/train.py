import torch 
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn as nn
import math
import numpy as np 
import cv2 
class Trainer():

    def __init__(self, gen , dis,data, fixed_noise,epochs = 50, batch_size = 32, device = 'cuda'):

        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print('Device Name:', torch.cuda.get_device_name(0))
            else:
                print('CUDA Device not available...')
                self.device = torch.device('cpu')
        
        self.generator = gen.to(self.device)
        self.discriminator = dis.to(self.device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.gen_optimizer = Adam(self.generator.parameters(), lr = 0.0002, betas = (0.5, 0.99))
        self.dis_optimizer = Adam(self.discriminator.parameters(), lr = 0.0002, betas=(0.5, 0.999))

        self.loss_criterion = nn.BCELoss()

        self.dataloader = data 
        self.z = fixed_noise 
    
    def plot_images(self, noise,epoch,generator):
        
        ## Generate image using Generator  
        with torch.no_grad():
            output_img = generator(noise).detach().cpu()
        
        bs = noise.size(0)
        nd = noise.size(1)
        ## Plot them
        for i in range(bs):
            if nd == 1:
                img = output_img[i].numpy()
                plt.subplot(int(math.sqrt(bs)), int(math.sqrt(bs)), i+1)
                plt.imshow(img[0], cmap='gray', interpolation='none')
                plt.set_axis('off')
                plt.tight_layout()
                plt.savefig('Epoch_{}.png'.format(epoch+1))
            elif nd == 3:
                img = np.transpose(output_img[i].numpy(), (1,2,0))
                plt.subplot(int(math.sqrt(bs)), int(math.sqrt(bs)), i+1)
                plt.imshow(img, interpolation='none')
                plt.set_axis('off')
                plt.tight_layout()
                plt.savefig('Epoch_{}.png'.format(epoch+1))
    
      def save_checkpoint(self, loss_value, epoch, mode = 'every_epoch'):

        if mode == 'best':
            if self.best_loss == 'na' :
                self.best_loss = loss_value
                torch.save(self.generator.state_dict(), 'Epoch_{}_Generator.pth'.format(epoch+1))
                print('Model Saved [Epoch:{}]'.format(epoch+1))
            else:
                if loss_value < self.best_loss:
                    self.best_loss = loss_value
                    torch.save(self.generator.state_dict(), 'Epoch_{}_Generator.pth'.format(epoch+1))
                    print('Model Saved [Epoch:{}]'.format(epoch+1))
        else:
            torch.save(self.generator.state_dict(), 'Epoch_{}_Generator.pth'.format(epoch+1))
            print('Model Saved [Epoch:{}]'.format(epoch+1))


    def train(self):

        self.gen_loss = []
        self.dis_loss = []
        
        for epoch in range(self.epochs):
            
            epoch_loss = {'gen_loss':[], 'dis_loss':[]}
            print('[Epoch: {} / {}]'.format(epoch+1, self.epochs))            
            
            for batch in tqdm(self.dataloader):
                
                images = batch.to(self.device)
                
                ## Retrive the Batch Size
                bs = images.size(0)
                nd = images.size(1)

                ## Labels for Real and Fake Images
                targetr = torch.ones(bs, device = self.device)
                targetz = torch.zeros(bs, device = self.device)

                ## Create noise
                noise = torch.randn(bs,nd,1,1).to(self.device)
                                                        
                # Discriminator Training
                self.discriminator.zero_grad()     

                dis_out = self.discriminator(images).squeeze()
                dis_real_loss = self.loss_criterion(dis_out, targetr)
                              
                gen_fake_out = self.generator(noise)
                dis_out = self.discriminator(gen_fake_out.detach()).squeeze()
                dis_fake_loss = self.loss_criterion(dis_out, targetz)
                loss_dis = dis_fake_loss + dis_real_loss
                loss_dis.backward()
                self.dis_optimizer.step()
                epoch_loss['dis_loss'].append(loss_dis.item())                

                ## Generator Training
                self.generator.zero_grad()        
                    
                ## Predictions by Discriminator for Fake Images
                dis_out = self.discriminator(gen_fake_out).squeeze()
                            
                ## Calculate the error for Generator
                gen_loss = self.loss_criterion(dis_out, targetr)
                gen_loss.backward()
                self.gen_optimizer.step()

                epoch_loss['gen_loss'].append(gen_loss.item())

            self.gen_loss.append(torch.mean(torch.FloatTensor(epoch_loss['gen_loss'])))
            self.dis_loss.append(torch.mean(torch.FloatTensor(epoch_loss['dis_loss'])))

            self.plot_images(self.z, epoch, self.generator)

            self.save_checkpoint(self.gen_loss[-1], epoch)

            print('')
            print('Generator Loss: {} | Discriminator Loss: {}'.format(self.gen_loss[-1],   self.dis_loss[-1]))
        