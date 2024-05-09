import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation

class SepConv(nn.Module):
    def __init__(self, in_channels: int , out_channels:int, kernel_size:int, stride=1, padding=0,depth_count:int=2, is_last=False ,growth_factor=1,groups:int=1) -> None:
        # depth_count >2
        super().__init__()
        growth=in_channels*growth_factor
        activation=nn.LeakyReLU(inplace=True)

        
        layers=[nn.Conv2d(in_channels=in_channels, out_channels=growth ,kernel_size=kernel_size,stride=stride,padding=padding, groups=in_channels, bias=False),
                nn.BatchNorm2d(growth),
                activation]
        for _ in range(1,depth_count-1):
            layers.extend([nn.Conv2d(in_channels=growth,out_channels=growth,kernel_size=3,stride=1,padding=1, groups=growth, bias=False),
                           nn.BatchNorm2d(growth),
                           activation])
            
        last_layer= [nn.Conv2d(in_channels=growth,out_channels=out_channels,kernel_size=1,groups=1, bias=True)]
        
        layers.extend(last_layer)   # this last one is pointwise
        
        self.conv=nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class MyModel1(nn.Module): #new
    def __init__(self, num_classes: int = 1000) -> None:

        super().__init__()


            
        self.Dconvs=nn.Sequential(
            nn.Conv2d(3,20,3,2,1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(inplace=True),
            SepConv(20,40,3,stride=2,padding=1,growth_factor=4 ,depth_count=3),# Avgpool-> 56 
            SepConv(40,40,3,stride=2,padding=1,growth_factor=3 ,depth_count=3),# then Avgpool -> 28 
            SepConv(40,80,3,stride=2,padding=1,growth_factor=4 ,depth_count=3), # then concate avgpool -> 14
            SepConv(80,80,3,stride=2,padding=1,growth_factor=3 ,depth_count=3),  # then concate Avgpool -> 7
            SepConv(80,num_classes,3,padding=1, growth_factor=num_classes//80,depth_count=3),
            # SepConv(160,320,3, padding=1, growth_factor=3,depth_count=2),
            ) # then concate avgpool -> 1
        




        self.fc=nn.Sequential(
        nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
        )
        
        



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.Dconvs(x)
        x=self.fc(x)
        return x

class MyModel2(nn.Module): #new
    def __init__(self, num_classes: int = 1000) -> None:

        super().__init__()


            
        self.Dconvs=nn.Sequential(
            nn.Conv2d(3,20,3,2,1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(inplace=True),
            SepConv(20,40,3,stride=2,padding=1,growth_factor=4 ,depth_count=3),# Avgpool-> 56 
            SepConv(40,40,3,stride=2,padding=1,growth_factor=3 ,depth_count=3),# then Avgpool -> 28 
            SepConv(40,80,3,stride=2,padding=1,growth_factor=4 ,depth_count=3), # then concate avgpool -> 14
            SepConv(80,80,3,stride=2,padding=1,growth_factor=3 ,depth_count=3),  # then concate Avgpool -> 7

            # SepConv(160,320,3, padding=1, growth_factor=3,depth_count=2),
            ) # then concate avgpool -> 1
        




        self.fc=nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(80, num_classes)
        )
        
        



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.Dconvs(x)
        x=self.fc(x)
        return x