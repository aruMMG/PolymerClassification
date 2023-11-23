import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd
import os



class IRDataset(Dataset):

    def __init__(self, directory):
        # Get the list of all subdirectories in the directory
        self.class_list = os.listdir(directory)
        self.data_list = []
        self.label_list = []
        
        # Loop over each subdirectory
        for i, class_name in enumerate(self.class_list):
            class_path = os.path.join(directory, class_name)
            file_list = os.listdir(class_path)
            
            # Read each CSV file in the subdirectory and extract the data
            for file in file_list:
                file_path = os.path.join(class_path, file)
                X = np.genfromtxt(file_path, delimiter=",")
                X = np.expand_dims(X, 0)
                self.data_list.append(X)
                self.label_list.append(float(i))
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        X = torch.tensor(self.data_list[index])
        y = torch.tensor(self.label_list[index])
        return X, y
    
class IRDatasetFromArrayFC(Dataset):

    def __init__(self, tup):
        # Get the list of all subdirectories in the directory
        self.tup = tup
        self.dataarrayX, self.dataarrayY, self.data_number = self.filter()
    def __len__(self):
        return len(self.dataarrayY)
    def __getitem__(self, index):
        X = torch.tensor(self.dataarrayX[index,:])
        y = torch.tensor(self.dataarrayY[index])
        d = torch.tensor(self.data_number[index])
        return X.float(), y.long(), d.int() 
    def filter(self):
        dataarrayX = None
        dataarrayY = None
        data_number = None
        for i, class_name in enumerate(self.tup):
            if dataarrayX is None:
                dataarrayX = class_name
                dataarrayY = np.ones(class_name.shape[0], dtype=float)*i
                data_number = np.arange(len(dataarrayY))
            else:
                dataarrayX = np.vstack([dataarrayX, class_name])
                dataarrayY = np.concatenate([dataarrayY, np.ones(class_name.shape[0], dtype=float)*i])
                data_number = np.concatenate([data_number, np.arange(len(data_number), len(data_number) + class_name.shape[0])])
        return dataarrayX, dataarrayY, data_number  
    
class IRDatasetFromArray(Dataset):

    def __init__(self, tup):
        # Get the list of all subdirectories in the directory
        self.tup = tup
        self.dataarrayX, self.dataarrayY, self.data_number = self.filter()
    def __len__(self):
        return len(self.dataarrayY)
    def __getitem__(self, index):
        X = torch.tensor(self.dataarrayX[index,:]).unsqueeze(dim=0)
        y = torch.tensor(self.dataarrayY[index])
        d = torch.tensor(self.data_number[index])
        return X.float(), y.long(), d.int()
    def filter(self):
        dataarrayX = None
        dataarrayY = None
        data_number = None
        for i, class_name in enumerate(self.tup):
            if dataarrayX is None:
                dataarrayX = class_name
                dataarrayY = np.ones(class_name.shape[0], dtype=float)*i
                data_number = np.arange(len(dataarrayY))
            else:
                dataarrayX = np.vstack([dataarrayX, class_name])
                dataarrayY = np.concatenate([dataarrayY, np.ones(class_name.shape[0], dtype=float)*i])
                data_number = np.concatenate([data_number, np.arange(len(data_number), len(data_number) + class_name.shape[0])])
            
        return dataarrayX, dataarrayY, data_number  
    
if __name__=="__main__":
        dataset = IRDataset("data")
        dataloader = DataLoader(dataset, batch_size=32,shuffle=True)
        for data, labels in dataloader:
            print(data.dtype)
            print(labels.dtype)