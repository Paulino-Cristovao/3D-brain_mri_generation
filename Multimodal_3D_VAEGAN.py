#
from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import math
#import torchio as tio
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
#import torchvision.models as models

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import Compose, RandFlip, Resize, ScaleIntensity,ToTensor, CenterSpatialCrop

#print(monai.__version__)
from loss import loss_function
import torchvision.models as models

parser = argparse.ArgumentParser(description='Pytorch DCGAN - Medical Data')
parser.add_argument('--workers',default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--num_epochs', default=10, type=int, metavar='epochs',
                    help='number of total epochs to run')
parser.add_argument('--ngpu', default=2, type=int, metavar='ngpu',
                    help='number of GPUs to run')
parser.add_argument('--channel', default=1, type=int, metavar='channel',
                    help='channel image ')
parser.add_argument('--features', default=32, type=int, metavar='features',
                    help='dimension of the feature map [Discriminator|Generator] ')
parser.add_argument('--batch_size', default=4, type=int,
                    metavar='batch_size', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--latent_z', default=256, type=int, metavar='latent_z',
                    help='number of latent dimension')
parser.add_argument('--beta1', default=0.5, type=int, metavar='beta1',
                    help='beta for Adam Optimizers')
parser.add_argument('-save', '--save', default=100, type=int, metavar='save',
                    help='save every epoch')
parser.add_argument('-plot_inter', '--plot_inter', default=50, type=int, metavar='save',
                    help='for plot')
parser.add_argument('--beta', default=1, type=int, metavar='beta',
                    help='beta value for loss function _ recons')
parser.add_argument('--gamma', default=1, type=int, metavar='gamma',
                    help='beta value for loss function _ kl')



root_save = '/I3M_IO_CALCULS_2/allInOne/3DVAEGAN_0000/'

# create folder for saving training and test results
global args
args = parser.parse_args()

if not os.path.isdir(str(args.features)):
    os.makedirs(root_save+str(args.features))

if not os.path.isdir(str(args.features)+'/'+str(args.num_epochs)):
    os.makedirs(root_save+str(args.features)+'/'+str(args.num_epochs))


# random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)

k = 2000 # save results at each k

# Device
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

# Data loader
#data_dir_train = '/I3M_IO_CALCULS_2/3D_240bigans/BraTS2021_3D50.npz'
data_dir_train = '/I3M_IO_CALCULS_2/allInOne/Data/BraTS2021_3D30.npz'

def Data(dir):
    data_load = np.load(dir)
    train_data = np.array(data_load['data'])
    
    return train_data


# Get training and test data
train_data = Data(data_dir_train)


class MyDataset(Dataset):
    def __init__(self,data,transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        imgs = self.data[index] 
        x0 = imgs[0]       
        x1 = imgs[1]
                
        if x0.max() != 0 and x1.max() != 0:
            x0 = x0.astype('float32').reshape((1, 240, 240, 155))
            x1 = x1.astype('float32').reshape((1, 240, 240, 155))

        if self.transform is not None:
            return self.transform(x0),self.transform(x1)
      
    def __len__(self):
        return len(self.data)


# Apply some data transformation
transform_A = Compose([
                        CenterSpatialCrop((180, 180, 155)),
                        Resize((128,128,128)),
                        ScaleIntensity(),
                        #RandFlip(prob=0.1, spatial_axis=None)
                        ])
                        
transform_mask = Compose([
                        CenterSpatialCrop((180, 180, 155)),
                        Resize((128,128,128)),
                        ])


class PerceptualLoss(nn.Module):
    def __init__(self,vgg16):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg16
        #vgg16[0] = nn.Conv2d(1, 64, 3, 1, 1)

    def forward(self, input2, target2):
        # Normalize input and target
        input = input2.view(-1,1,128,128)
        target = target2.view(-1,1,128,128)

        input_norm = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
        target_norm = (target - torch.min(target)) / (torch.max(target) - torch.min(target))
        
        # Compute feature maps for input and target images
        input_features = self.vgg(input_norm)
        target_features = self.vgg(target_norm)

        # Compute mean squared error (MSE) between feature maps
        loss = nn.MSELoss()(input_features, target_features)
        return loss


vgg16 = models.vgg16(pretrained=True)
vgg16.features[0] = nn.Conv2d(1, 64, 3, 1, 1)

vgg16.eval()
# Freeze all model parameters
for param in vgg16.parameters():
    param.requires_grad_(False)



# train data
print("Size of the data:",len(train_data))
dataset_train = MyDataset(train_data, transform=transform_A)
train_x = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True)


# test data
dataset_test = MyDataset(train_data, transform=transform_mask)
test_x = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=False)



# Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    #if classname.find('Conv')!= -1:
    #	nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class View(nn.Module):
    def __init__(self,size):
        super(View,self).__init__()
        self.size = size
    
    def forward(self, tensor):
        return tensor.view(self.size)


# reconstruction loss
def reconstruction_loss(x, x_recon):
    recon_loss = nn.MSELoss(size_average=None, reduction="mean")

    return recon_loss(x_recon, x)

# kl divergence
def kl_divergence(mu, logvar):
    latent_kl = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()).mean(dim=0)
    total_kld = latent_kl.sum()

    return total_kld


def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu



# Encoder
class Encoder(nn.Module):
    def __init__(self,channel,features):
        super(Encoder, self).__init__()
        self.channel = channel 
        self.features = features

        padding = (0,0,0)
        padding1 = (1,1,1)

        self.conv1 = nn.Conv3d(self.channel, self.features,4,2,padding1, bias=False)
        self.conv2 = nn.Conv3d(self.features, self.features*2,4,2,padding1, bias=False)
        self.conv3 = nn.Conv3d(self.features*2, self.features*3,4,2,padding1, bias=False)
        self.conv4 = nn.Conv3d(self.features*3, self.features*4,4,2,padding1, bias=False)
        #self.conv5 = nn.Conv3d(self.features, args.features,4,2,padding1, bias=False)

        self.conv5_ = nn.Conv3d(args.features*2*4, args.features*5,4,2,padding1, bias=False)
        self.conv6 = nn.Conv3d(args.features*5, args.latent_z,4,2,padding1, bias=False) #3,1

        self.batchNorm1 = nn.BatchNorm3d(self.features, momentum=0.9)
        self.batchNorm2 = nn.BatchNorm3d(self.features*2, momentum=0.9)
        self.batchNorm3 = nn.BatchNorm3d(self.features*3, momentum=0.9)
        self.batchNorm4 = nn.BatchNorm3d(self.features*4, momentum=0.9)
        self.batchNorm5 = nn.BatchNorm3d(self.features*5, momentum=0.9)
        self.batchNorm6 = nn.BatchNorm3d(args.latent_z, momentum=0.9)

        self.view = View((-1, args.latent_z*2*2))

        self.linear1 = nn.Linear(args.latent_z*2*2,args.latent_z*2*2)
        #self.linear2 = nn.Linear(args.latent_z,args.latent_z*2*2)

        #self.batchNorm5 = nn.BatchNorm1d(args.latent_z)
        #self.batchNorm6 = nn.BatchNorm1d(args.latent_z*2)

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,x1):
        x = self.relu(self.batchNorm1(self.conv1(x)))   #8
        x = self.relu(self.batchNorm2(self.conv2(x)))   #16
        x = self.relu(self.batchNorm3(self.conv3(x)))   #32
        x = self.relu(self.batchNorm4(self.conv4(x)))  #64
        #x = self.relu(self.batchNorm3(self.conv5(x)))  #128

        x1 = self.relu(self.batchNorm1(self.conv1(x1))) #8
        x1 = self.relu(self.batchNorm2(self.conv2(x1))) #16
        x1 = self.relu(self.batchNorm3(self.conv3(x1))) #32
        x1 = self.relu(self.batchNorm4(self.conv4(x1))) #64
        #x1 = self.relu(self.batchNorm3(self.conv5(x1))) #128

        x_= self.relu(self.batchNorm5(self.conv5_(torch.concat((x,x1),dim=1))))
        
        x = self.relu(self.batchNorm6(self.conv6(x_)))   #256
        x = self.view(x)    
        x = self.linear1(x)

        return x



# Generator Code
class Generator(nn.Module):
    def __init__(self,channel,latent_z,features):
        super(Generator, self).__init__()
        self.latent_z = latent_z
        self.features = features
        self.channel = channel

        padding = (0,0,0)
        padding1 = (1,1,1)

        self.linear = nn.Linear(self.latent_z,self.latent_z*1*1*1, bias=False)

        self.view = View((-1, self.latent_z,1,1,1))
        
        self.convT1 = nn.ConvTranspose3d(self.latent_z, self.features*5,4,1,padding,bias=False)     #4
        self.convT2 = nn.ConvTranspose3d(self.features*5, self.features*4,4,2,padding1,bias=False)  #8
        self.convT3 = nn.ConvTranspose3d(self.features*4, self.features*3,4,2,padding1,bias=False)  #16
        self.convT4 = nn.ConvTranspose3d(self.features*3, self.features*2,4,2,padding1,bias=False)  #32
        self.convT5 = nn.ConvTranspose3d(self.features*2, self.features,4,2,padding1,bias=False)  #64
        #self.convT6 = nn.ConvTranspose3d(self.features, self.features,4,2,padding1,bias=False)    #128
        self.convT7 = nn.ConvTranspose3d(self.features, self.channel,4,2,padding1,bias=False)       #256

        self.batchNorm1 = nn.BatchNorm3d(self.features*5, momentum=0.9)
        self.batchNorm2 = nn.BatchNorm3d(self.features*4, momentum=0.9)
        self.batchNorm3 = nn.BatchNorm3d(self.features*3, momentum=0.9)
        self.batchNorm4 = nn.BatchNorm3d(self.features*2, momentum=0.9)
        self.batchNorm5 = nn.BatchNorm3d(self.features, momentum=0.9)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)

      
    def forward(self,x):
        x = self.linear(x)
        x = self.view(x)
        x = self.relu(self.batchNorm1(self.convT1(x)))
        x = self.relu(self.batchNorm2(self.convT2(x)))
        x = self.relu(self.batchNorm3(self.convT3(x)))
        x = self.relu(self.batchNorm4(self.convT4(x)))
        x = self.relu(self.batchNorm5(self.convT5(x)))
        #x = self.relu(self.batchNorm3(self.convT6(x)))
        x = self.tanh(self.convT7(x))
        

        return x


# Generator Code
class VAE_Generator(nn.Module):
    def __init__(self,channel,latent_z,features):
        super(VAE_Generator, self).__init__()
        self.latent_z = latent_z
        self.features = features
        self.channel = channel
        self.enc = Encoder(args.channel, args.features)
        self.view = View((-1, args.latent_z*2*2*2))
        self.decoder = Generator(args.channel,args.latent_z,args.features)

        self.fc_mu = nn.Linear(args.latent_z*2*2*2,args.latent_z)
        self.fc_logvar = nn.Linear(args.latent_z*2*2*2,args.latent_z)

        
    def encode(self,x,x1):
        x = self.enc(x,x1)
        x = self.view(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
        

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z
    
    def add_noise(self,z, noise_multiplier=0.01):
        noise = torch.randn_like(z) * noise_multiplier

        return z + noise

    def decode(self,x):
        x = self.decoder(x)
        
        return x

      
    def forward(self,x,x1):
        mu, logvar = self.encode(x,x1)
        z = self.reparameterize(mu, logvar)
        #z = self.add_noise(z)
        x_recon = self.decode(z)

        return x_recon, mu, logvar, z


# Create the generator and initialize # Multiple GPU
generator = VAE_Generator(args.channel,args.latent_z,args.features).to(device)
generator = nn.DataParallel(generator)
generator.apply(weights_init)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channel,features):
        super(Discriminator, self).__init__()
        self.channel = channel
        self.features = features

        padding = (0,0,0)
        padding1 = (1,1,1)

        self.conv1 = nn.Conv3d(self.channel, self.features,4,2,padding1,bias=False)     #4
        self.conv2 = nn.Conv3d(self.features, self.features*2,4,2,padding1,bias=False)    #8
        self.conv3 = nn.Conv3d(self.features*2, self.features*3,4,2,padding1,bias=False)    #16
        self.conv4 = nn.Conv3d(self.features*3, self.features*4,4,2,padding1,bias=False)    #32
        self.conv5 = nn.Conv3d(self.features*4, self.features*5,4,2,padding1,bias=False)    #64
        #self.conv6 = nn.Conv3d(self.features, self.features,4,2,padding1,bias=False)   #128
        self.conv7 = nn.Conv3d(self.features*5, 1,2,4,padding,bias=False)                 #256

        self.batchNorm1 = nn.BatchNorm3d(self.features, momentum=0.9)
        self.batchNorm2 = nn.BatchNorm3d(self.features*2, momentum=0.9)  
        self.batchNorm3 = nn.BatchNorm3d(self.features*3, momentum=0.9)
        self.batchNorm4 = nn.BatchNorm3d(self.features*4, momentum=0.9)  
        self.batchNorm5 = nn.BatchNorm3d(self.features*5, momentum=0.9)

        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2)
       

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.batchNorm2(self.conv2(x)))
        x = self.leakyrelu(self.batchNorm3(self.conv3(x)))
        x = self.leakyrelu(self.batchNorm4(self.conv4(x)))
        x = self.leakyrelu(self.batchNorm5(self.conv5(x)))
        #x = self.leakyrelu(self.batchNorm1(self.conv6(x)))
        #x = self.conv7(x)
        #x = x.view(x.size()[0],-1)
        #x = self.sigmoid(x)
        x = self.sigmoid(self.conv7(x))

        return x


# create the Discriminator
discriminator = Discriminator(args.channel,args.features).to(device)
discriminator = nn.DataParallel(discriminator)
discriminator.apply(weights_init)


generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)

l2_loss = nn.MSELoss().to(device)
perceptual_loss = PerceptualLoss(vgg16).to(device)
criterion = nn.BCELoss().to(device)


generator.train()
discriminator.train()

# Training Loop
G_losses = []
D_losses = []

losses = []
losses_ = []

rec_noise = []
noise_noise = []

# for later plot
output_train = []
output_test  = []

fixed_noise = torch.randn(args.batch_size,args.latent_z, device=device)

# Start training
for epoch in range(args.num_epochs):
    for i, data in enumerate(train_x, 0):
        # Move data to device
        real_images = data[0].to(device)
        real_images1 = data[1].to(device)

        # Update discriminator
        discriminator.zero_grad()

        # Train with real images
        real_labels = torch.ones(real_images.size(0),).to(device)
        real_outputs = discriminator(real_images).view(-1)

        #print(real_labels.shape, real_outputs.shape)
        discriminator_real_loss = criterion(real_outputs, real_labels)

        # Train with fake images
        z = torch.randn(real_images.size(0), args.latent_z).to(device)

        x_recon, mu, logvar, z_prime = generator(real_images,real_images1)
        
        fake_images = generator.module.decode(z)

        #tumor_loss = l2_loss(fake_images*real_images1,real_images*real_images1)

        fake_labels = torch.zeros(real_images.size(0),).to(device)
        recons_outputs = discriminator(x_recon.detach()).view(-1)
        discriminator_recons_loss = criterion(recons_outputs, fake_labels)

        fake_outputs = discriminator(fake_images.detach()).view(-1)
        discriminator_fake_loss = criterion(fake_outputs, fake_labels)

        # Compute total discriminator loss and update parameters
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss + discriminator_recons_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Update generator
        generator.zero_grad()

        # Train with fake images again
        fake_outputs = discriminator(fake_images).view(-1)
        generator_loss = criterion(fake_outputs, real_labels) #+ tumor_loss

        # Compute VAE loss
        #x_recon, mu, logvar, z_prime = generator(real_images,real_images1)

        #vae_recon_loss = l2_loss(x_recon, real_images)
        #print(real_images.shape, x_recon.shape)

        vae_recon_loss = perceptual_loss(real_images, x_recon)
        vae_kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = args.gamma*vae_recon_loss + args.beta*vae_kl_loss

        # Compute perceptual loss
        perceptual_loss_real = perceptual_loss(real_images, x_recon)
        perceptual_loss_fake = perceptual_loss(real_images, fake_images)
        perceptual_loss_total = perceptual_loss_real + perceptual_loss_fake

        # Compute total generator loss and update parameters
        #generator_loss += 0.01 * perceptual_loss_total
        generator_loss += vae_loss + perceptual_loss_total
        generator_loss.backward()
        generator_optimizer.step()


        # Save losses for plotting later
        G_losses.append(generator_loss.item())
        D_losses.append(discriminator_loss.item())

        
        if math.isnan(generator_loss) or math.isnan(discriminator_loss):
            print("The find nan for one of these loss generator_loss or discriminator_loss: ",generator_loss.item(),discriminator_loss.item())
            break


        # Output training stats
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{args.num_epochs}] Batch [{i}/{len(train_x)}] Discriminator Loss: {discriminator_loss.item():.6f} Generator Loss: {generator_loss.item():.6f}")

    if (epoch % k == 0 and epoch > 0):
        output_train.append([epoch,real_images.cpu().detach().numpy(),fake_images.cpu().detach().numpy(),x_recon.cpu().detach().numpy()])

        ######################## compute loss values ###################
        loss_ssim, loss_psrn, loss_mse = loss_function(real_images.cpu().detach().numpy(),fake_images.cpu().detach().numpy())
        losses.append([epoch, loss_ssim, loss_psrn, loss_mse])

    # check the performance of the generator on fixed_noise
    if epoch % k == 0 and epoch > 0:
      with torch.no_grad():
        fake = generator.module.decode(fixed_noise)
        output_test.append([epoch,real_images.cpu().detach().numpy(),fake.cpu().detach().numpy()])
        #losses_.append([epoch, loss_ssim_, loss_psrn_, loss_mse_])

    # save results to plot later
    if (epoch % k == 0 and epoch > 0):
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/train_out'+'_'+str(epoch)+'.npz', data = output_train, num_epochs = args.num_epochs, allow_pickle=True)
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/test_out'+'_'+str(epoch)+'.npz', data = output_test, num_epochs = args.num_epochs, allow_pickle=True)
        ####################### save losses ############################
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/G_D_losses.npz', G_losses=G_losses, D_losses=D_losses,allow_pickle=True)

        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/loss_function_train.npz', losses=losses, allow_pickle=True)
        #np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/loss_function_test.npz', losses=losses_, allow_pickle=True)

        
        plt.plot(G_losses, label="G-loss",color='r')
        plt.plot(D_losses, label="D-loss",color='b')
        plt.legend()
        plt.title("Generator and Discriminator Loss")
        plt.grid()
        plt.savefig(root_save+str(args.features)+'/'+str(args.num_epochs)+'/lossGD.png')
        plt.clf()

############# THE END  ############## 