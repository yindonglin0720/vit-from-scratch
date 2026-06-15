import torch
from torchvision import datasets, transforms

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_data=datasets.MNIST(root='./data',train=True,transform=transform,download=True)