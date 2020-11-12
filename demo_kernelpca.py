import os.path
import argparse
import numpy as np
from tqdm import tqdm
import torch
import _io
import streamlit as st
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
#import yacs
import os
from scipy.io import loadmat
from PIL import Image

device = torch.device('cuda:0')
torch.cuda.set_device(0)
## utility to handle latent code state
GAN_HASH_FUNCS = {
    _io.TextIOWrapper : id,
torch.nn.backends.thnn.THNNFunctionBackend:id,
torch.nn.parameter.Parameter:id,
torch.Tensor:id,
#yacs.config.CfgNode:id,
#_thread.lock:id,
#builtins: id,
#builtins.PyCapsule:id,
}


@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS,show_spinner=False)
def get_comp():
    npys = np.load('./kernel_components.npz')
    components = npys['components']
    mean_ = npys['mean_']
    var_ = npys['var_']
    return components, mean_, var_

components, mean_, var_ = get_comp()

steps = [None]*16
for i in range(16):
    steps[i] = st.sidebar.slider('Component'+str(i), 0., 1.0,0.0, 0.02)

base_kernel = components[0]*steps[0]
for i in range(15):
    base_kernel += components[i+1]*steps[i+1]
print(np.linalg.norm(components[0]))
print(np.sum(base_kernel))
base_kernel = (base_kernel)+mean_#/ (np.sum(base_kernel)+0.000000001)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
tempkernel = np.reshape(base_kernel,(21,21))
ax.imshow(tempkernel,cmap='gnuplot2',vmin=-1.0, vmax=1.0)

st.write(fig)