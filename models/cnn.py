import torch
from torch import nn

class CNN(nn.module):
    def __init__(self,dropout:float=0.0,use_batchnorm:bool=False,
                 in_channels=1,num_classes:int=10)
        super().__init__()

        c1,c2= 5,10
        layers=[]
        layers += [nn.Conv2d(in_channels,c1,kernel_size=3,padding=1),
                   nn.ReLU(inplace=True)]
        if use_batchnorm:
            layers += [nn.BatchNorm2d(c1)]
        layers += [nn.MaxPool2d(2,2)]
        layers += [nn.Conv2d(in_channels,c1,c2,kernel_size=3,padding=1),
                   nn.ReLU(inplace=True)]
        if use_batchnorm:
            layers += [nn.BatchNorm2d(c2)]
        layers += [nn.MaxPool2d(2,2)]
        self.feauters = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(7*7*c2,num_classes)
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x=self.feauters(x)
        x=torch.flatten(x,1)
        x=self.dropout(x)
        x=self.classifier(x)

        return x 
    