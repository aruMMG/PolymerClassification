import torch
import torchvision

# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
import numpy as np
from dataset import IRDatasetFromArray
from model import SpectroscopyTransformerEncoder, SpectroscopyTransformerEncoder_PreT, InceptionNetwork, InceptionNetwork_PreT
from utils import createFolders, plot9Data, save_arguments_to_file

import os
import csv

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def visualise_pre(model, data, save_path):
    pass

def plot_pair(data, pre_out, savename="example2"):
    y = np.arange(4000)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot data1 in the first subplot
    ax1.plot(y, data, label='Data', color='blue')
    ax1.set_ylabel('Data')
    ax1.set_title('Subplots of Data 1 and Data 2')

    # Plot data2 in the second subplot
    ax2.plot(y, pre_out, label='pre_out', color='red')
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('pre_out')
    plt.tight_layout()
    # fig, ax = plt.subplots(4,4,figsize=(15,15))
    # ax[0,0].plot(y,data)
    # ax[0,1].plot(y,pre_out)
    fig.savefig(savename+".jpg")


if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    model_weights_path = "logFile/exp6/weights_incep_pre/best_0.pt"
    # model = SpectroscopyTransformerEncoder(num_classes=3, mlp_size=64)
    model = InceptionNetwork_PreT(num_classes=3)
    # l = torch.load(model_weights_path)['model']
    # print(l.keys())
    model.load_state_dict(torch.load(model_weights_path)['model'])
    model = model.to(device=device)
    model = model.float()
    model.pre.register_forward_hook(get_activation('pre'))
    name_HDPE = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/LDPE_baseline.npy")
    for i in range(5):
        data = name_HDPE[i,:]
        data = torch.tensor(data).unsqueeze(dim=0)
        print(data.shape)
        data = data.float()
        print(data.dtype)
        data = data.to(device=device)
        model.eval()
        out = model(data)
        pre_out = activation['pre']
        pre_out = pre_out.squeeze()
        data = data.squeeze()
        data = data.to(device='cpu')
        pre_out = pre_out.to(device='cpu')
        # pre_out = pre_out.unsqueeze(dim=0)
        plot_pair(data.numpy(), pre_out.numpy(), savename=f"PrePlot/preIncep_exp6_LDPE_{i}_baseline")
    name_HDPE = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/HDPE_baseline.npy")
    for i in range(5):
        data = name_HDPE[i,:]
        data = torch.tensor(data).unsqueeze(dim=0)
        print(data.shape)
        data = data.float()
        print(data.dtype)
        data = data.to(device=device)
        model.eval()
        out = model(data)
        pre_out = activation['pre']
        pre_out = pre_out.squeeze()
        data = data.squeeze()
        data = data.to(device='cpu')
        pre_out = pre_out.to(device='cpu')
        # pre_out = pre_out.unsqueeze(dim=0)
        plot_pair(data.numpy(), pre_out.numpy(), savename=f"PrePlot/preIncep_exp6_HDPE_{i}_baseline")
    name_HDPE = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/PET_baseline.npy")
    for i in range(5):
        data = name_HDPE[i,:]
        data = torch.tensor(data).unsqueeze(dim=0)
        print(data.shape)
        data = data.float()
        print(data.dtype)
        data = data.to(device=device)
        model.eval()
        out = model(data)
        pre_out = activation['pre']
        pre_out = pre_out.squeeze()
        data = data.squeeze()
        data = data.to(device='cpu')
        pre_out = pre_out.to(device='cpu')
        # pre_out = pre_out.unsqueeze(dim=0)
        plot_pair(data.numpy(), pre_out.numpy(), savename=f"PrePlot/preIncep_exp6_PET_{i}_baseline")