import os
import shutil
import glob
import subprocess
from moviepy.editor import VideoFileClip
import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from PIL import Image
# Let's see if we have an available GPU
import numpy as np
import random


# Seed random generator for repeatibility  
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

im_size=256
every_s=0.2
audio_sr=8000
emotions = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
    }
def setup_env():
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("GPU available")
    else:
        print("GPU *NOT* available. Will use CPU (slow)")

    # Seed random generator for repeatibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Download data if not present already
    if not os.path.exists("data/metadata.csv"):
        prepare_data()
    # compute_mean_and_std()
    # Make checkpoints subdir if not existing
    os.makedirs("checkpoints", exist_ok=True)


# def prapare_data():
#     data_folder = get_data_location()
#     for em in emotions.values():
#         os.makedirs(data_folder + "/" + em, exist_ok=True)

#     # Using glob.glob to get a list of files that match a pattern
#     for path in glob.glob('**/01-*.mp4', recursive=True):
#         vid_dir=data_folder+"/"+emotions[path[-18:-16]]+path[-25:-4]
#         os.makedirs(vid_dir, exist_ok=True)
#         subprocess.call(["ffmpeg", "-i", path,"-vf",f'fps={fps}, scale=256:256', f"{vid_dir}/image_%04d.jpg"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
#         subprocess.call(["ffmpeg", "-y", "-i", path, f"{vid_dir}/audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def prepare_data():
    data_folder = get_data_location()


    df = pd.DataFrame(columns=["video_name","mel_spec_path","image_path", "class"])
    # Using glob.glob to get a list of files that match a pattern
    files=glob.glob('**/01-*.mp4', recursive=True)
    for path in tqdm(files,total=len(files),desc="preparing data", ncols=80):
        vid_dir=data_folder+"/"+emotions[path[-18:-16]]+path[-25:-4]
        os.makedirs(vid_dir+"/images", exist_ok=True)
        os.makedirs(vid_dir+"/mel-spec", exist_ok=True)
        subprocess.call(["ffmpeg", "-y", "-i", path, f"{vid_dir}/raw_audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        clip=VideoFileClip(path).resize(width=im_size,height=im_size)
        time_stamps=np.arange(0,clip.duration, every_s)
        for i in range(len(time_stamps)-1):
            image_path=vid_dir+"/images/"+f"00{i}.jpeg"
            mel_spec_path=vid_dir+"/mel-spec/"+f"00{i}.npy"
            subclip=clip.subclip(t_start=time_stamps[i],t_end=time_stamps[i+1])
            subclip.save_frame(image_path)
            raw_sound=subclip.audio.to_soundarray(fps=audio_sr)
            mel_spec=librosa.feature.melspectrogram(y=raw_sound, sr=audio_sr).squeeze(axis=2)
            log_mel_spectrogram = librosa.power_to_db(mel_spec).astype(np.float32)
            np.save(mel_spec_path,log_mel_spectrogram)
            
            #df.append({"video_name" : path[-25:-4], "mel_spec_path":mel_spec_path,"image_path":image_path, "class": emotions[path[-18:-16]]}, ignore_index=True)
            df.loc[len(df)] = [path[-25:-4], mel_spec_path,image_path,  emotions[path[-18:-16]]]
    
    df.to_csv(data_folder+"/metadata.csv")



    


def get_data_location():
    """
    Find the location of the dataset
    """
    

    if os.path.exists("data"):
        data_folder = "data"



    else:
        raise IOError("Please download the dataset first")

    return data_folder








# Compute image normalization
def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    # cache_file = "mean_and_std.pt"
    # if os.path.exists(cache_file):
    #     print(f"Reusing cached mean and std")
    #     d = torch.load(cache_file)

    #     return d["mean"], d["std"]

    # folder = get_data_location()
    # ds = datasets.ImageFolder(
    #     folder, transform=transforms.Compose([transforms.ToTensor()])
    # )
    # dl = torch.utils.data.DataLoader(
    #     ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    # )

    # mean = 0.0
    # for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
    #     batch_samples = images.size(0)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    # mean = mean / len(dl.dataset)

    # var = 0.0
    # npix = 0
    # for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
    #     batch_samples = images.size(0)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    #     npix += images.nelement()

    # std = torch.sqrt(var / (npix / 3))

    # # Cache results so we don't need to redo the computation
    # torch.save({"mean": mean, "std": std}, cache_file)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    return mean, std


def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    if group_name.lower() == "loss":
        ax.set_ylim([None, None])


def replace_insatance(model,replaced, new):
    for name,module in model.named_children():
        if isinstance(module, replaced):
            setattr(model,name,new)
        else:
            replace_insatance(module,replaced, new)


def plot_confusion_matrix(pred, truth):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)

    fig, sub = plt.subplots(figsize=(14, 12))
    with sns.plotting_context("notebook"):
        idx = (confusion_matrix == 0)
        confusion_matrix[idx] = np.nan
        sns.heatmap(confusion_matrix, annot=True, ax=sub, linewidths=0.5, linecolor='lightgray', cbar=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def load_model(model_name, model):

    try:
        checkpoint = torch.load(f'checkpoints/last_{model_name}.pt')
    except FileNotFoundError:
        try:
            checkpoint = torch.load(f'checkpoints/best_{model_name}.pt')
        except FileNotFoundError:
            print("New wheigts are initilaized!!")
            return 1

    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Previously trained model weights state_dict loaded...')
    epochs = checkpoint['epochs']
    print(f"Previously trained for {epochs} number of epochs...")

    return epochs +1

