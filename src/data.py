import math
import torch
from pathlib import Path
from torchvision import datasets
import multiprocessing
import numpy as np
from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
from torch.utils.data import default_collate



def get_data_loaders(
    batch_size: int = 32, num_classes: int =1000, num_workers: int = -1, limit: int = -1
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

    # YOUR CODE HERE:
    # create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    # HINT: resize the image to 256 first, then crop them to 224, then add the
    # appropriate transforms for that step
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

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms["train"]
    )
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = datasets.ImageFolder(
        base_path / "val",
        transform=data_transforms["valid"]
    )
   
    # prepare data loaders
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    

    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        #sampler=sampler,
     collate_fn=collate_fn,
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
