import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
from utils import save_indexes_to_file, asls_dataset

    
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



class IRDatasetFromNames(Dataset):

    def __init__(self, x,y,n, FC=False, raw=False):
        self.dataarrayX = x
        self.dataarrayY = y
        self.data_number = n
        self.FC = FC
        self.raw = raw
    def __len__(self):
        return len(self.dataarrayY)
    def __getitem__(self, index):
        if self.FC:
            X = torch.tensor(self.dataarrayX[index,:])
        else:
            if self.raw:
                X = torch.tensor(self.dataarrayX[index,:]).unsqueeze(dim=0)
            else:
                X = torch.tensor(self.dataarrayX[index,:]).unsqueeze(dim=0)
        y = torch.tensor(self.dataarrayY[index])
        d = self.data_number[index]
        # d = torch.tensor(self.data_number[index])
        return X.float(), y.long(), d
    

def split_data(args):
    dataarrayX_train, dataarrayY_train, data_number_train, dataarrayX_val, dataarrayY_val, data_number_val = None, None, None, None, None, None
    index_list_train, index_list_val = [], []
    index_list_test = []
    dataarrayY_test, dataarrayX_test, data_number_test = None, None, None
    for i, data_name in enumerate(args.data_names):
        data = np.load(os.path.join(args.data_dir, data_name+".npy"))

        if args.baseline:
            print("Using baseline")
            data = asls_dataset(data)
        index_list_train.append(data_name)
        first_subset, second_subset, first_subset_indexes, second_subset_indexes = split_array_with_percentage(data, args.train_perc)
        index_list_train.extend(first_subset_indexes)
        first_subset_indexes = np.array([f"{data_name}{index}" for index in first_subset_indexes])
        if dataarrayX_train is None:
            dataarrayX_train = first_subset
            dataarrayY_train = np.ones(len(first_subset), dtype=float)*i
            data_number_train = first_subset_indexes
        else:
            dataarrayX_train = np.vstack([dataarrayX_train, first_subset])
            dataarrayY_train = np.concatenate([dataarrayY_train, np.ones(len(first_subset), dtype=float)*i])
            data_number_train = np.concatenate([data_number_train, first_subset_indexes])

        if args.test_perc > 0:
            index_list_test.append(data_name)
            percentage = (args.test_perc / (args.test_perc + args.val_perc)) * 100
            first_subset, second_subset, first_subset_indexes, second_subset_indexes = split_array_with_percentage(second_subset, percentage)
            index_list_test.extend(first_subset_indexes)
            first_subset_indexes = np.array([f"{data_name}{index}" for index in first_subset_indexes])
            if dataarrayX_test is None:
                dataarrayX_test = first_subset
                dataarrayY_test = np.ones(len(first_subset), dtype=float)*i
                data_number_test = first_subset_indexes
            else:
                dataarrayX_test = np.vstack([dataarrayX_test, first_subset])
                dataarrayY_test = np.concatenate([dataarrayY_test, np.ones(len(first_subset), dtype=float)*i])
                data_number_test = np.concatenate([data_number_test, first_subset_indexes])

        index_list_val.append(data_name)
        index_list_val.extend(second_subset_indexes)
        second_subset_indexes = np.array([f"{data_name}{index}" for index in second_subset_indexes])
        if dataarrayX_val is None:
            dataarrayX_val = second_subset
            dataarrayY_val = np.ones(len(second_subset), dtype=float)*i
            data_number_val = second_subset_indexes
        else:
            dataarrayX_val = np.vstack([dataarrayX_val, second_subset])
            dataarrayY_val = np.concatenate([dataarrayY_val, np.ones(len(second_subset), dtype=float)*i])
            data_number_val = np.concatenate([data_number_val, second_subset_indexes])
    
    os.mkdir(f'logFile/{args.log_name}/data_split/{args.fold_num}')
    save_indexes_to_file(index_list_train, f'logFile/{args.log_name}/data_split/{args.fold_num}/index_train.txt')
    save_indexes_to_file(index_list_val, f'logFile/{args.log_name}/data_split/{args.fold_num}/index_val.txt')
    if args.test_perc > 0:
        save_indexes_to_file(index_list_test, f'logFile/{args.log_name}/data_split/{args.fold_num}/index_test.txt')
    return dataarrayX_train, dataarrayY_train, data_number_train, dataarrayX_test, dataarrayY_test, data_number_test, dataarrayX_val, dataarrayY_val, data_number_val

def split_array_with_percentage(arr, percentage):
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")

    # Calculate the number of elements to include in the first subset
    num_elements = len(arr)
    num_first_subset = int(num_elements * (percentage / 100))

    # Create a list of indexes to shuffle
    indexes = list(range(num_elements))
    random.shuffle(indexes)

    # Split the indexes into two subsets
    first_subset_indexes = indexes[:num_first_subset]
    second_subset_indexes = indexes[num_first_subset:]

    # Create the actual subsets based on the selected indexes
    first_subset = np.array([arr[i] for i in first_subset_indexes])
    second_subset = np.array([arr[i] for i in second_subset_indexes])

    return first_subset, second_subset, first_subset_indexes, second_subset_indexes

def dataset_from_txt(args, txt_file):
    # Create an empty list to store the combined subsets
    combined_subsets = []
    dataarrayY = None
    data_number = None
    y_label = 0

    # Read the TXT file with names and indexes
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # Initialize variables to keep track of the current subset
    current_name = None
    current_subset = []
    name_list = []

    # Iterate through each line in the TXT file
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            if line.isdigit():
                current_subset.append(int(line))
            else:
                # If a new name is encountered, load the corresponding .npy file
                if current_name:
                    npy_file = os.path.join(args.data_dir, current_name + '.npy')
                    if os.path.exists(npy_file):
                        array = np.load(npy_file)
                        subset = [array[i] for i in current_subset]
                        combined_subsets.extend(subset)
                        if dataarrayY is None:
                            dataarrayY = np.ones(len(subset), dtype=float)*y_label
                            data_number = np.array(current_subset)
                            y_label += 1
                        else:
                            dataarrayY = np.concatenate([dataarrayY, np.ones(len(subset), dtype=float)*y_label])
                            data_number = np.concatenate([data_number, np.array(current_subset)])
                            y_label += 1
                current_name = line
                name_list.append(line)
                current_subset = []

    # Load and append the last subset
    if current_name:
        npy_file = os.path.join(args.data_dir, current_name + '.npy')
        if os.path.exists(npy_file):
            array = np.load(npy_file)
            subset = [array[i] for i in current_subset]
            combined_subsets.extend(subset)
            if dataarrayY is None:
                dataarrayY = np.ones(len(subset), dtype=float)*y_label
                data_number = np.array(current_subset)
                y_label += 1
            else:
                dataarrayY = np.concatenate([dataarrayY, np.ones(len(subset), dtype=float)*y_label])
                data_number = np.concatenate([data_number, np.array(current_subset)])
                y_label += 1
    if args.baseline:
        print("Using baseline")
        subset = asls_dataset(np.array(combined_subsets))
    else:
        subset = np.array(combined_subsets)
    return subset, dataarrayY, data_number, name_list

if __name__=="__main__":
        data_x = np.random.random((8,4000))
        data_y = np.random.randint(0,2,size=(8))
        data_num = np.array(["HDPE1","HDPE2","HDPE3","HDPE4","HDPE5","HDPE6","HDPE7","HDPE8",])
        
        dataset = IRDatasetFromNames(data_x, data_y, data_num, FC=True)
        dataloader = DataLoader(dataset, batch_size=2,shuffle=True)
        for data, labels, data_num in dataloader:
            print(data.dtype)
            print(labels.dtype)
