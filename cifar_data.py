
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split,DataLoader
import os


data_dir = os.path.join(os.getcwd(),"data")

if os.path.isdir(os.path.join(data_dir,"cifar100")):
    PATH_TO_CIFAR100 = os.path.join(data_dir,"cifar100")
else:
    os.makedirs(os.path.join(data_dir,"cifar100"))
    PATH_TO_CIFAR100 = os.path.join(data_dir,"cifar100")

# PATH_TO_CIFAR100 = "/mnt/769EC2439EC1FB9D/vsc_projs/cifar100"

transforms = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Resize(224),
    v2.Normalize(mean=(0.5071, 0.4867, 0.4408),std=(0.2675, 0.2565, 0.2761)),    #cifar100 norm values...

    # augments if to be added
    v2.AutoAugment()

])


cifar100 = datasets.CIFAR100(root=PATH_TO_CIFAR100,download=True,transform=transforms)
train_data, val_data = random_split(dataset=cifar100, lengths=[45000,5000])     #90/10


def get_cifar_train_loader(batch_size,shuffle=True,num_workers=4):
    return DataLoader(dataset=train_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)

def get_cifar_val_loader(batch_size,shuffle=True,num_workers=4):
    return DataLoader(dataset=val_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)




