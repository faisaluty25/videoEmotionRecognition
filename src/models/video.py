import torch
import torch.nn as nn

from .audio import Model1
from .image import get_model_transfer_learning

class Model1(nn.Module): #new
    def __init__(self, num_classes: int = 1000) -> None:

        super().__init__()


            
        self.im_model = get_model_transfer_learning('shufflenet_v2_x1_0', with_head=False)
        self.audio = Model1(with_head=False)
        


        self.avgpool=nn.AdaptiveAvgPool2d(1)

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, num_classes)
        )
        
        



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        im=self.avgpool(self.im_model(x[0]))
        audio=self.avgpool(self.audio(x[1]))
        im_au=torch.cat((im, audio), dim=1)
        x=self.fc(x)
        return x