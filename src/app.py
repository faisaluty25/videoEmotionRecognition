import streamlit as st
import numpy as np
import torchvision.transforms as T
import torch
from PIL import Image
import io
import pandas as pd
def classify(image):

    # Load image that has been uploaded
    fn = io.BytesIO(image.getvalue())

    img = Image.open(fn)
    img.load()


    # Display the image
    ratio = img.size[0] / img.size[1]
    c = img.copy()
    c.thumbnail([ratio * 200, 200])
    st.image(c)

    # Transform to tensor
    img=img.resize((224,224),resample=Image.BILINEAR, reducing_gap=1)
    img=np.array(img, dtype=np.float32)
    img=np.transpose(img[None,],[0,3,1,2])

    timg = torch.Tensor(img)

    # Calling the model
    softmax = learn_inf(timg).cpu().numpy()
    
    # Get the indexes of the classes ordered by softmax
    # (larger first)
   
    
    st.write(softmax[0])

learn_inf = torch.jit.load('checkpoints/original_exported.pt')

uploaded_images = st.file_uploader("Upload your images here...",type=['png','jpg', 'jpeg'],accept_multiple_files=True)

for image in uploaded_images:
    classify(image)

