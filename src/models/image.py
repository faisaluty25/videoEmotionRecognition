import torch
import torch.nn as nn
import torchvision
import torchvision.models as t_models
from torchvision.ops import SqueezeExcitation



def get_model_transfer_learning(model_name="resnet18", n_classes=50):

    # Get the requested architecture
    if hasattr(t_models, model_name):

        model_transfer = getattr(t_models, model_name)(pretrained=True)

    else:

        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")
    
    # detrmine the last layer
    classifier='fc' if hasattr(model_transfer,'fc') else 'classifier'  
    
    # detrmine num_features( this not a good way but it works)
    try:
        setattr(model_transfer,classifier, nn.Linear(2,1))
        x=torch.zeros(1,3,224,224)
        model_transfer(x)
    except Exception as e:
        error=str(e)
        start_index=error.index('(')
        end_index=error[start_index:].index('a')
        in_features=int(error[start_index:start_index+end_index].split('x')[1])



    # Freeze all parameters in the model
    # HINT: loop over all parameters. If "param" is one parameter,
    # "param.requires_grad = False" freezes it
    # for param in model_transfer.parameters():
    #     param.requires_grad = False
    



    # Add the linear layer at the end with the appropriate number of classes
    # 1. get numbers of features extracted by the backbone
    num_ftrs  = in_features

    # 2. Create a new linear layer with the appropriate number of inputs and
    #    outputs
    fc=nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs,n_classes))
               
        
    setattr(model_transfer,classifier, fc)

    return model_transfer

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