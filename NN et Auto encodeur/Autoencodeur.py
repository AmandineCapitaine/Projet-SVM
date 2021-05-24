from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
from keras.datasets import mnist
import class_ae as ae

batch_size = 64

use_cuda = torch.cuda.is_available()
#use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': batch_size}

if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)




train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2,**test_kwargs)
    


model = ae.Autoencoder().to(device)
distance = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), eps=1e-8, lr = 0.0022) lr pas ouf
optimizer = optim.Adadelta(model.parameters())
num_epochs = 15

print("started")
for epoch in range(num_epochs):
    for batch_idx, (data,_) in enumerate(train_loader):
        img = data.to(device)
        if use_cuda:
            img = Variable(img).cuda()
        else:
            img = Variable(img).cpu()
        output = model(img)
        #print(output.shape, img.shape)
        loss = distance(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

torch.save(model.state_dict(), "mnist_cnn.pt")

#torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': loss}, C:\Users\teaca\Documents\INSA\3A\3MIC S2\MODELISATION)