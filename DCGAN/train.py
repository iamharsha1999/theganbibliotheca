import torch 
from tqdm import tqdm 
import matplotlib.pyplot as plt

## Function for Plotting the image every epoch
def plot_images(noise,epoch,generator):

  ## Generate image using Generator  
  output_img = generator(noise).cpu().detach()
  bs = noise.size(0)
  ## Plot them
  for i in range(bs):
    img = output_img[i].numpy()
    plt.subplot(int(math.sqrt(bs)), int(math.sqrt(bs)), i+1)
    plt.imshow(img[0], cmap='gray', interpolation='none')
    plt.savefig('Epoch_{}.png'.format(epoch+1))

## Train Function
def train(dataloader, discriminator, generator, optimizer_gen, optimizer_dis, criterion, device ='gpu', epochs=100):

    """
        dataloader         ->   Batched Data
        discriminator     ->   Discriminator Network
        generator           ->   Generator Network
        device                  ->   CPU or GPU (By default GPU)
        optimizer_gen  ->   Optimizer for Generator
        optimizer_dis    ->   Optimizer for Discriminator
        criterion              ->   Loss function 

    """
    g_loss = []
    d_loss = []
    fixed_noise = torch.randn(16,100).view(-1,100,1,1).to(device)

    if device == 'gpu':
        ## CUDA Devie Initialization
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Device Name:', torch.cuda.get_device_name(0))
        else:
            device = torch.device('cpu')
            print('CUDA device not available')
    else:
            device = torch.device('cpu')


    for epoch in range(epochs):
        
        epoch_loss = {'gen_loss':[], 'dis_loss':[]}
        print('[Epoch: {} / {}]'.format(epoch+1, epochs))
        
        for batch in tqdm(dataloader):
            
            images,_ = batch
            images = images.to(device)
            
            ## Retrive the Batch Size
            bs = images.size(0)

            ## Create noise
            noise = torch.randn(bs,100,1,1).to(device)

            ## Labels for Real Images
            targetr = torch.ones(bs, device = device)

            ## Labels for Fake Images
            targetz = torch.zeros(bs, device = device)
                    
            ## Clear the accumulated gradients
            discriminator.zero_grad()
                        
            ## Predictions by Discriminator for Real Images        
            dis_out = discriminator(images).squeeze()
                                
            ## Calculate the error for Real Images
            dis_real_loss = criterion(dis_out, targetr)
            dis_real_loss.backward()
            
            ## Generate images from noise
            gen_fake_out = generator(noise)
            
            ## Labels for Fake Images
            targetz = torch.zeros(bs, device = device)
            
            ## Predictions by Discriminator for Fake Images   
            dis_out = discriminator(gen_fake_out.detach()).squeeze()
                              
            ## Calculate the error for Fake Images
            dis_fake_loss = criterion(dis_out, targetz)
            dis_fake_loss.backward()
            
            ## Update the EPOCH Loss
            epoch_loss['dis_loss'].append((dis_real_loss + dis_fake_loss).item())
            
            ## Update the parameters
            optimizer_dis.step()
            
            ## Clear the accumulated gradients
            generator.zero_grad()        
                
            ## Predictions by Discriminator for Fake Images
            dis_out = discriminator(gen_fake_out).squeeze()
                        
            ## Calculate the error for Generator
            gen_loss = criterion(dis_out, targetr)
            gen_loss.backward()
            
            ## Update the parameters
            optimizer_gen.step()
            
            epoch_loss['gen_loss'].append(gen_loss.item())

        g_loss.append(torch.mean(torch.FloatTensor(epoch_loss['gen_loss'])))
        d_loss.append(torch.mean(torch.FloatTensor(epoch_loss['dis_loss'])))

        plot_images(fixed_noise,epoch, generator)

        print('')
        print('Generator Loss: {} | Discriminator Loss: {}'.format(g_loss[-1], d_loss[-1]))
        