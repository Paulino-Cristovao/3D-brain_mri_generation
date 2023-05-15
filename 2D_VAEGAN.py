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
#import torchio as tio
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math 
from loss import loss_function


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
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='batch_size', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--latent_z', default=1024, type=int, metavar='latent_z',
                    help='number of latent dimension')
parser.add_argument('--beta1', default=0.05, type=int, metavar='beta1',
                    help='beta for Adam Optimizers')
parser.add_argument('-save', '--save', default=100, type=int, metavar='save',
                    help='save every epoch')
parser.add_argument('-plot_inter', '--plot_inter', default=50, type=int, metavar='save',
                    help='for plot')
parser.add_argument('--beta', default=10, type=int, metavar='beta',
                    help='beta value for loss function ')
parser.add_argument('--gamma', default=20, type=int, metavar='gamma',
                    help='beta value for loss function')


# create folder for saving training and test results
global args
args = parser.parse_args()

root_save = '2DVAEGAN_1input/'

if not os.path.isdir(str(args.features)):
	os.makedirs(root_save+str(args.features))

if not os.path.isdir(str(args.features)+'/'+str(args.num_epochs)):
	os.makedirs(root_save+str(args.features)+'/'+str(args.num_epochs))


# random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)


k = 2000

# Device
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

# Data loader
data_dir_train = 'BraTS2021.npz'

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

    def normalize(self,data):
        img_min = np.min(data)

        data_ = (data - img_min) / (np.max(data) - img_min)  

        return data_

    def __getitem__(self, index):
        imgs = self.data[index] 
        #x0 = imgs[0]/2313.0
        x0 = self.normalize(imgs[0])
        #x1 = self.normalize(imgs[1]/4.0)
        
        x0 = x0.astype('float32').reshape((240, 240, 1))
        #x1 = x1.astype('float32').reshape((240, 240, 1))
        
        if self.transform is not None:
          return self.transform(image=x0)["image"] #,self.transform(image=x1)["image"]
      
    def __len__(self):
        return len(self.data)


# Apply some data transformation
transform_A = A.Compose([
                A.CenterCrop(height=180, width=180),
                A.Resize(height=128, width=128),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
                    
])


transform_mask = A.Compose([
            A.CenterCrop(height=180, width=180),
            A.Resize(height=128, width=128),
            ToTensorV2(),
           
        ])

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
	if classname.find('Conv')!= -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
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


# Generator Code
class Generator(nn.Module):
    def __init__(self,channel,latent_z,features):
        super(Generator, self).__init__()
        self.latent_z = latent_z
        self.features = features
        self.channel = channel

        padding = 0
        padding1 = 1

        self.linear = nn.Linear(self.latent_z,self.latent_z*1*1, bias=False)

        self.view = View((-1, self.latent_z,1,1))
        
        self.convT1 = nn.ConvTranspose2d(self.latent_z, self.features*8,4,1,padding,bias=False)     #4
        self.convT2 = nn.ConvTranspose2d(self.features*8, self.features*8,4,2,padding1,bias=False)  #8
        self.convT3 = nn.ConvTranspose2d(self.features*8, self.features*8,4,2,padding1,bias=False)  #16
        self.convT4 = nn.ConvTranspose2d(self.features*8, self.features*8,4,2,padding1,bias=False)  #32
        self.convT5 = nn.ConvTranspose2d(self.features*8, self.features*8,4,2,padding1,bias=False)  #64
        #self.convT6 = nn.ConvTranspose2d(self.features*8, self.features*8,4,2,padding1,bias=False)    #128
        self.convT7 = nn.ConvTranspose2d(self.features*8, self.channel,4,2,padding1)                  #256

        self.batchNorm1 = nn.BatchNorm2d(self.features*8)
        #self.batchNorm2 = nn.BatchNorm2d(self.features*8)
        #self.batchNorm3 = nn.BatchNorm2d(self.features*4)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

      
    def forward(self,x):
        x = self.linear(x)
        x = self.view(x)

        x = self.relu(self.batchNorm1(self.convT1(x)))
        x = self.relu(self.batchNorm1(self.convT2(x)))
        x = self.relu(self.batchNorm1(self.convT3(x)))
        x = self.relu(self.batchNorm1(self.convT4(x)))
        x = self.relu(self.batchNorm1(self.convT5(x)))
        #x = self.relu(self.batchNorm3(self.convT6(x)))
        x = self.tanh(self.convT7(x))

        return x

# Create the generator and initialize # Multiple GPU
netG = Generator(args.channel,args.latent_z,args.features).to(device)
netG = nn.DataParallel(netG)

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.02.
netG.apply(weights_init)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channel,features):
        super(Discriminator, self).__init__()
        self.channel = channel
        self.features = features

        padding = 0
        padding1 = 1

        self.conv1 = nn.Conv2d(self.channel, self.features*8,4,2,padding1,bias=False)       #4
        self.conv2 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1,bias=False)    #8
        self.conv3 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1,bias=False)    #16
        self.conv4 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1,bias=False)    #32
        self.conv5 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1,bias=False)   #64
        #self.conv6 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1,bias=False)  #128
        self.conv7 = nn.Conv2d(self.features*8, 1,2,4,padding,bias=False)                  #256

        self.batchNorm1 = nn.BatchNorm2d(self.features*8)
        #self.batchNorm2 = nn.BatchNorm2d(self.features*16)  
        #self.batchNorm3 = nn.BatchNorm2d(self.features*16)
        
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.20,inplace=True)       

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.batchNorm1(self.conv2(x)))
        x = self.leakyrelu(self.batchNorm1(self.conv3(x)))
        x = self.leakyrelu(self.batchNorm1(self.conv4(x)))
        x = self.leakyrelu(self.batchNorm1(self.conv5(x)))
        #x = self.leakyrelu(self.batchNorm3(self.conv6(x)))
        x = self.sigmoid(self.conv7(x))

        return x

# create the Discriminator
netD = Discriminator(args.channel,args.features).to(device)
netD = nn.DataParallel(netD)
netD.apply(weights_init)



# Encoder
class Encoder(nn.Module):
    def __init__(self,channel,features):
        super(Encoder, self).__init__()
        self.channel = channel 
        self.features = features

        padding = 0
        padding1 = 1

        self.conv1 = nn.Conv2d(self.channel, self.features*8,4,2,padding1, bias=False)
        self.conv2 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1, bias=False)
        self.conv3 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1, bias=False)
        self.conv4 = nn.Conv2d(self.features*8, self.features*8,4,2,padding1, bias=False)
        self.conv5 = nn.Conv2d(self.features*8, args.features*8,4,2,padding1, bias=False)

        self.conv5_ = nn.Conv2d(args.features*8, args.features*8,4,2,padding1, bias=False)        
        self.conv6 = nn.Conv2d(args.features*8, args.latent_z,4,2,padding1, bias=False) #3,1

        self.batchNorm1 = nn.BatchNorm2d(self.features*8)
        #self.batchNorm2 = nn.BatchNorm2d(self.features*8)
        #self.batchNorm3 = nn.BatchNorm2d(args.features*4)
        self.batchNorm4 = nn.BatchNorm2d(args.latent_z)

        self.view = View((-1, args.latent_z*2*2))

        self.linear1 = nn.Linear(args.latent_z*2*2,args.latent_z)
        self.linear2 = nn.Linear(args.latent_z,args.latent_z*2)

        self.relu = nn.LeakyReLU(0.20,inplace=True)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self,x):
        x = self.relu(self.batchNorm1(self.conv1(x)))   #8
        x = self.relu(self.batchNorm1(self.conv2(x)))   #16
        x = self.relu(self.batchNorm1(self.conv3(x)))   #32
        x = self.relu(self.batchNorm1(self.conv4(x)))  #64
        #x = self.relu(self.batchNorm1(self.conv5(x)))  #128

        ##x1 = self.relu(self.batchNorm1(self.conv1(x1))) #8
        ##x1 = self.relu(self.batchNorm2(self.conv2(x1))) #16
        ##x1 = self.relu(self.batchNorm2(self.conv3(x1))) #32
        #x1 = self.relu(self.batchNorm2(self.conv4(x1))) #64
        #x1 = self.relu(self.batchNorm3(self.conv5(x1))) #128

        #x_= self.relu(self.batchNorm3(self.conv5_(torch.concat((x,x1),dim=1))))
        x_= self.relu(self.batchNorm1(self.conv5_(x)))
        
        x = self.relu(self.batchNorm4(self.conv6(x_)))   #256
        x = self.view(x)    

        x = self.relu(self.linear1(x)) 
        x = self.sigmoid(self.linear2(x))

        mean = x[:,:args.latent_z]
        logvar = x[:,args.latent_z:]

        z = reparameterize(mean,logvar)

        return x, mean, logvar, z


# Create Encoder
netE = Encoder(args.channel, args.features).to(device)
netE = nn.DataParallel(netE)
netE.apply(weights_init)

# Initialization BCE function
criterion_D =  nn.BCELoss().to(device) 
criterion_G =  nn.BCELoss().to(device)
criterion_E =  nn.BCELoss().to(device)
criterion_l1 =  nn.L1Loss().to(device)

# create a batch of latent vectors that
# we will use to visualize the progression of the generator
def fixed_noise():
    return  torch.randn(args.batch_size,args.latent_z, device=device)
#fixed_noise = torch.randn(args.batch_size,args.latent_z device=device)

#def noise(b_size):
#    return torch.randn(b_size, args.latent_z, device=device)

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

# Optimizers for both Generator and Discriminator
optimD = optim.Adam(netD.parameters(), lr=0.0001)
optimG = optim.Adam(netG.parameters(), lr=0.0001)
optimE = optim.Adam(netE.parameters(), lr=0.0001)

# Training Loop
G_losses = []
D_losses = []
E_losses = []

losses = []
losses_ = []

# for later plot
output_train = []
output_test  = []

netG.train()
netD.train()
netE.train()

    
print("Sarting training loop ... ")
# for each epoch
for epoch in range(args.num_epochs):
    # for each batch in dataloader
    for i, data in enumerate(train_x, 0):
        # Train with all-real batch
        real_cpu = data.to(device, non_blocking=True, dtype=torch.float32) 
        #real_cpu1 = data[1].to(device, non_blocking=True, dtype=torch.float32)

        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float32, device=device)
        label1 = torch.full((b_size,), real_label, dtype=torch.float32, device=device)
        label2 = torch.full((b_size,), real_label, dtype=torch.float32, device=device)
        label3 = torch.full((b_size,), real_label, dtype=torch.float32, device=device)

        _,mean,logvar,z_noise = netE(real_cpu)   

        x_recons = netG(z_noise) 
        noise = torch.randn(b_size, args.latent_z, device=device)

        x_noise  = netG(noise)
        
        ######## Train Discriminator ######
        optimD.zero_grad()

        d_real_loss = criterion_D(netD(real_cpu).view(-1),label.clone())

        label1.fill_(fake_label)
        d_fake_loss = criterion_D(netD(x_noise).view(-1),label1)

        d_recon_loss = criterion_D(netD(x_recons).view(-1),label1)

        dis_loss = d_recon_loss + d_real_loss + d_fake_loss
        dis_loss.backward(retain_graph=True)

        optimD.step()

        ##########  Train Generator ########
        optimG.zero_grad()

        output1 = netD(real_cpu).view(-1)
        label2.fill_(real_label)
        d_real_loss1 = criterion_G(output1, label2)

        output2 = netD(x_recons).view(-1)
        label3.fill_(fake_label)
        d_recon_loss1 = criterion_D(output2,label3)

        output3 = netD(x_noise).view(-1)
        d_fake_loss1 = criterion_G(output3,label3)

        d_img_loss = d_real_loss1 + d_recon_loss1 + d_fake_loss1
        gen_img_loss = - d_img_loss

        rec_loss = reconstruction_loss(real_cpu,x_recons)
        #rec_loss = ((x_recons - real_cpu)**2).mean()

        err_dec = args.gamma * rec_loss + gen_img_loss

        err_dec.backward(retain_graph=True)
        #optimG.step()

        ############ Train Encoder #####################
        #prior_loss = kl_divergence(mean, logvar)
        #err_enc = prior_loss + rec_loss * args.beta

        prior_loss = 1+logvar-mean.pow(2) - logvar.exp()
        prior_loss = (-0.5*torch.sum(prior_loss))/torch.numel(mean.data)

        err_enc = prior_loss + rec_loss * args.beta

        netE.zero_grad()
        err_enc.backward()

        #optimD.step()
        optimG.step()
        optimE.step()
        
        # output training stats
        if i%500 == 0:
          print(f'[Epoch/num_epochs]: {epoch}/{args.num_epochs},| errD:{dis_loss.item()} | errE:{err_enc.item()}, | errDe:{err_dec.item()}')

        # Save losses for plotting later
        G_losses.append(err_dec.item())
        D_losses.append(dis_loss.item())
        E_losses.append(err_enc.item())

        if math.isnan(dis_loss.item()) or math.isnan(err_dec.item()):
            print("The find nan for one of these loss generator_loss or discriminator_loss: ",err_dec.item(),dis_loss.item())
            break
        
    
    if (epoch % k == 0 and epoch > 0):
        output_train.append([epoch,real_cpu.cpu().detach().numpy(),x_noise.cpu().detach().numpy(),x_recons.cpu().detach().numpy()])

        ######################## compute loss values ###################
        loss_ssim, loss_psrn, loss_mse = loss_function(real_cpu.cpu().detach().numpy(),x_noise.cpu().detach().numpy())
        losses.append([epoch, loss_ssim, loss_psrn, loss_mse])

    # check the performance of the generator on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise()).detach().cpu()
            output_test.append([epoch,real_cpu.cpu().detach().numpy(),fake.cpu().detach().numpy()])
            #loss_ssim_, loss_psrn_, loss_mse_ = loss_function(real_cpu.cpu().detach().numpy(),fake.cpu().detach().numpy())
            #losses_.append([epoch, loss_ssim_, loss_psrn_, loss_mse_])

    # save results to plot later
    if (epoch % k == 0 and epoch > 0):
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/train_out'+'_'+str(epoch)+'.npz', data = output_train, num_epochs = args.num_epochs, plot_inter =args.plot_inter, allow_pickle=True)
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/test_out'+'_'+str(epoch)+'.npz', data = output_test, num_epochs = args.num_epochs, plot_inter =args.plot_inter, allow_pickle=True)
        ####################### save losses ############################
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/G_E_D_losses.npz', G_losses=G_losses, D_losses=D_losses, E_losses=E_losses, allow_pickle=True)
        
        np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/loss_function_train.npz', losses=losses, allow_pickle=True)
        #np.savez(root_save+str(args.features)+'/'+str(args.num_epochs)+'/loss_function_test.npz', losses=losses_, allow_pickle=True)

        plt.plot(G_losses, label="G-loss",color='r')
        plt.plot(D_losses, label="D-loss",color='b')
        plt.legend()
        plt.title("Generator and Discriminator Loss")
        plt.grid()
        plt.savefig(root_save+str(args.features)+'/'+str(args.num_epochs)+'/lossGD.png')
        plt.clf()

        plt.plot(E_losses, label="E-loss",color='g')
        plt.legend()
        plt.title("Encoder Loss")
        plt.grid()
        plt.savefig(root_save+str(args.features)+'/'+str(args.num_epochs)+'/lossE.png')
        plt.clf()

############# THE END  ##############     
