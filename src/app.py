import gradio as gr
import numpy as np
import torch
from PIL import Image

def classify(image):
    # Transform to tensor
    img = image.resize((232, 232), resample=Image.BILINEAR, reducing_gap=1)
    img = np.array(img, dtype=np.float32)
    img = np.transpose(img[None, :], [0, 3, 1, 2])
    timg = torch.Tensor(img)

    # Calling the model
    softmax = learn_inf(timg).cpu().numpy()
    confidences = {learn_inf.class_names[i]: pre for i,pre in enumerate(softmax[0])}
    return confidences

# Load the model
learn_inf = torch.jit.load('checkpoints/original_exported.pt')

# Gradio interface
iface = gr.Interface(fn=classify,
                     inputs= gr.Image(type="pil"),
                     outputs=gr.Label(num_top_classes=3),
                     examples=[["data/Calm/01-01-02-01-01-01-01/images/000.jpeg"]],
                     title="Emotion Detector",
                     description="Upload an image to classify.")
iface.launch()


