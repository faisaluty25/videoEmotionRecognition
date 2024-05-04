import os

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms.functional as F
import torchvision.transforms as T
from .helpers import get_data_location


class Predictor(nn.Module):

    def __init__(self, model, class_names, mean:torch.Tensor, std: torch.Tensor):
        super().__init__()

        self.model = model.eval()
        self.class_names = class_names
        self.sig=nn.Sigmoid()
        self.mean=mean.mul_(255).view(-1,1,1)
        self.std=std.mul_(255).view(-1,1,1)
        # We use nn.Sequential and not nn.Compose because the former
        # is compatible with torch.script, while the latter isn't


    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            # 1. apply transforms
            x =x[:,:,16:16+224,16:16+224]
            x.sub_(self.mean).div_(self.std) 
            # 2. get the logits
            x  = self.model(x)
            # 3. apply softmax
            x  = self.sig(x)

            return x

# class Predictor(nn.Module):

#     def __init__(self, model, class_names, mean, std):
#         super().__init__()

#         self.model = model.eval()
#         self.class_names = class_names
#         self.sig=nn.Sigmoid()

#         # We use nn.Sequential and not nn.Compose because the former
#         # is compatible with torch.script, while the latter isn't
#         self.transforms = nn.Sequential(
#             T.Resize([256, ]),  # We use single int value inside a list due to torchscript type restrictions
#             T.CenterCrop(224),
#             T.ConvertImageDtype(torch.float),
#             T.Normalize(mean.tolist(), std.tolist())
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         with torch.no_grad():
#             # 1. apply transforms
#             x  =self.transforms(x) 
#             # 2. get the logits
#             x  = self.model(x)
#             # 3. apply softmax
#             #    HINT: remmeber to apply softmax across dim=1
#             x  =self.sig(x)

#             return x

def predictor_test(test_dataloader, model_reloaded):
    """
    Test the predictor. Since the predictor does not operate on the same tensors
    as the non-wrapped model, we need a specific test function (can't use one_epoch_test)
    """

    folder = get_data_location()
    test_data = datasets.ImageFolder(os.path.join(folder, "test"), transform=T.ToTensor())

    pred = []
    truth = []
    for x in tqdm(test_data, total=len(test_dataloader.dataset), leave=True, ncols=80):
        sigmoid = model_reloaded(x[0].unsqueeze(dim=0)).squeeze()
        pred.append(int(sigmoid>0.5))
        truth.append(int(x[1]))

    pred = np.array(pred)
    truth = np.array(truth)

    print(f"Accuracy: {(pred==truth).sum() / pred.shape[0]}")

    return truth, pred



