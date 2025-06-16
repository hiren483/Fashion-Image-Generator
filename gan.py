import torch
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_df = pd.read_csv("fashion-mnist_train.csv")

labels = torch.tensor(train_df['label'].values,dtype=torch.long)
features = torch.tensor(train_df.drop('label',axis = 1).values,dtype= torch.float32)/255.0

class FashionMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        # Return number of samples
        return len(self.labels)

    def __getitem__(self, i):
        # Return one sample at index idx
        image = self.features[i].reshape(1, 28, 28)  # reshape to 1x28x28 for CNN
        label = self.labels[i]
        return image, label

train_dataset = FashionMNISTDataset(features, labels)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

input_size = 784  
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size =64
learning_rate = 0.01

z_dim = 100
noise = torch.randn(batch_size, z_dim)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 128 * 7 * 7)
        
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1
        )
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        out = self.fc(x)               # [batch, 128*7*7]
        out = out.view(-1, 128, 7, 7) # reshape to [batch, 128, 7, 7]
        
        out = self.deconv1(out)       # [batch, 64, 14, 14]
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.deconv2(out)       # [batch, 1, 28, 28]
        out = self.tanh(out)          # pixel values between -1 and 1
        
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
            #input - [batch, 1, 28, 28]
        self.l1 = nn.Conv2d(in_channels=1 , out_channels=64,kernel_size=4,stride=2,padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.l2 = nn.Conv2d(in_channels=64 , out_channels=128 , kernel_size=4,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 7 * 7,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        out = self.l1(x)          # [batch, 64, 14, 14]
        out = self.leaky_relu(out)
        
        out = self.l2(out)        # [batch, 128, 7, 7]
        out = self.bn1(out)
        out = self.leaky_relu(out)
        
        out = out.view(out.size(0), -1)  # flatten: [batch, 128*7*7]
        out = self.fc(out)                # [batch, 1]
        out = self.sigmoid(out)           # output probability
        
        return out
    
import torch.optim as optim
import torch.nn as nn

# Initialize models
G = Generator().to(device)
D = Discriminator().to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
lr = 0.0002
betas = (0.5, 0.999)
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=betas)
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=betas)

num_epochs = 30
latent_dim = 100

for epoch in range(num_epochs):
    for real_images, _ in train_loader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        ### Train Discriminator ###
        optimizerD.zero_grad()
        
        # 1. Forward real images
        outputs = D(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()
        
        # 2. Forward fake images
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_images = G(noise)
        outputs = D(fake_images.detach())  # detach to avoid G gradient update here
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        
        d_loss = d_loss_real + d_loss_fake
        optimizerD.step()
        
        ### Train Generator ###
        optimizerG.zero_grad()
        
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_images = G(noise)
        outputs = D(fake_images)
        
        # Generator wants discriminator to output 1 for fake images
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizerG.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")



import matplotlib.pyplot as plt

# Set model to eval mode
G.eval()

