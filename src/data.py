import math
import torch
from pathlib import Path
from torchvision import datasets
import multiprocessing
import numpy as np
import pandas as pd
from .helpers import compute_mean_and_std, get_data_location, seed, every_s, audio_sr
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
from torch.utils.data import default_collate
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import os

class  RAVDESSDataset(torch.utils.data.Dataset):
    def __init__(self,df, is_image=True, is_mel=True,transforms=None):
        self.df=df
        self.is_image = is_image
        self.is_mel = is_mel
        self.transforms = transforms




    def __getitem__(self, idx):
        # load images 
        target = self.df.iloc[idx]['target']
        if self.is_image:
            img_path = self.df.iloc[idx]['image_path']
            img = read_image(img_path)
            
            if self.transforms:
                img = self.transforms(img)
            if not self.is_mel:
                return img, target


        


        if self.is_mel:
            mel_spec_path = self.df.iloc[idx]['mel_spec_path']
            mel_spec = np.load(mel_spec_path)
            if not self.is_image:
                return mel_spec[:int(every_s*audio_sr)][None,], target

        

        data=[img, mel_spec[:int(every_s*audio_sr)][None,]]
        # print(data.shape)

        return data, target

    def __len__(self):
        return len(self.df)

def get_data_loaders(
    batch_size: int = 32, num_classes: int =1000,valid_size=0.2, num_workers: int = -1, limit: int = -1, is_mel=True, is_image=True
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    # We will fill this up later
    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    train_size=200
    valid_size=230
    cutmix = T.CutMix(alpha=1.0 ,num_classes=num_classes)
    mixup = T.MixUp(alpha=0.2 ,num_classes=num_classes)
    cutmix_or_mixup = T.RandomChoice([cutmix, mixup])
    data_transforms = {
        "train": T.Compose([
            T.ToImage(),
            # T.RandomVerticalFlip(p=0.3),
            # T.RandomRotation(30),

            T.Resize((train_size+2,train_size+2), antialias=True),
            #T.RandomResizedCrop(train_size, antialias=True),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
            #T.RandAugment(num_ops=3),
            #T.ColorJitter(),
            #T.RandomPosterize(bits=1),
            #T.RandomAffine(degrees=(1, 70), translate=(0.1, 0.3), scale=(0.6, 1)),
            #T.RandomPerspective(distortion_scale=0.3, p=0.5),
            #T.RandomAutocontrast(),
            T.CenterCrop(train_size),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),# the mean and std are already scaled(as helper.py) no need to rescale them
            T.RandomErasing(p=0.1),
            


        ]),
        "valid": T.Compose([
            T.ToImage(),
            T.Resize((valid_size+2,valid_size+2),antialias=True),
            T.CenterCrop(valid_size),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
            
        ]),
        "test": T.Compose([
            T.ToImage(),
            # T.Resize(230),
            # T.CenterCrop(224),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ]),
    }
    df = pd.read_csv('data/metadata.csv',dtype={'class':'category'})
    df['target']=df['class'].cat.codes.astype(np.int64)
    train_df, valid_df = train_test_split(df,
                                stratify=df['class'],test_size=valid_size,random_state=seed)


    #  Create train and validation datasets
    train_data = RAVDESSDataset(train_df,is_image=is_image, is_mel=is_mel, transforms=data_transforms['train'])
    valid_data = RAVDESSDataset(valid_df,is_image=is_image, is_mel=is_mel, transforms=data_transforms['valid'])
   
    # prepare data loaders
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    

    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        #sampler=sampler,
     #collate_fn=collate_fn,
#        pin_memory=True,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    # YOUR CODE HERE:
    # obtain one batch of training images
    # First obtain an iterator from the train dataloader
    dataiter  = iter(data_loaders['train'])
    # Then call the .next() method on the iterator you just
    # obtained
    images, labels = next(dataiter)

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # YOUR CODE HERE:
    # Get class names from the train data loader
    class_names  = data_loaders['train'].dataset.classes

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(class_names[labels[idx].item()])

def visualize_image(image,label, class_names):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """


    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    image = invTrans(image)


    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    image = torch.permute(image, (1, 2, 0)).clip(0, 1)

    # Custom (rgb) grid color
    grid_color =torch.tensor([0, 0, 0])
    grid_size=32
    # Modify the image to include the grid
    image[:, ::grid_size, :] = grid_color
    image[::grid_size, :, :] = grid_color 


    plt.imshow(image)
    # print out the correct label for each image
        # .item() gets the value contained in a Tensor
    plt.title(class_names[label.item()])
