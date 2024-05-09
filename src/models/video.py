import torch
import torch.nn as nn

from .audio import Model1
from .image import get_model_transfer_learning

class Modelv(nn.Module): #new
    def __init__(self, num_classes: int = 1000) -> None:

        super().__init__()


            
        self.im_model = get_model_transfer_learning('shufflenet_v2_x1_0', n_classes=80)
        self.audio = Model1(with_head=False)
        


        self.avgpool=nn.AdaptiveAvgPool2d(1)

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, num_classes)
        )
        
        



    def forward(self, x) -> torch.Tensor:
        # (img, audio)
        # print(x[0].shape)
        x[0]=self.im_model(x[0])
        x[1]=self.avgpool(self.audio(x[1])).flatten(start_dim=1)
        im_au=torch.cat((x[0], x[1]), dim=1)
        return self.fc(im_au)
         