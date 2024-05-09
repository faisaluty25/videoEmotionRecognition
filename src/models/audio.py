import torch
import torch.nn as nn
from .image import SepConv
class Model1(nn.Module): #new
    def __init__(self, with_head=True ,num_classes: int = 1000) -> None:

        super().__init__()


            
        self.Dconvs=nn.Sequential(
            nn.Conv2d(1,20,3,2,1),
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
        ) if with_head else nn.Identity()
        
        



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.Dconvs(x)
        x=self.fc(x)
        return x