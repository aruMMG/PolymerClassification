import torch
import torchvision

# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
import numpy as np
from dataset import IRDatasetFromNames, dataset_from_txt
from model_ablation import SpectroscopyTransformerEncoder, SpectroscopyTransformerEncoder_PreT, InceptionNetwork, InceptionNetwork_PreT, FCNet, PSDNResNet
from utils import plot9Data, save_arguments_to_file, load_args_from_txt

import os
import csv

def test(args, model_class, test_dataloader, model_weights_path, n_classes=2, arch="trans_pre"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create an instance of the model
    if arch=="incep" or arch=="incep_avg" or arch=="res" or arch=="FC":
        model = model_class(num_classes=n_classes).to(device)
    else:
        model = model_class(input_size=args.input_size, num_classes=n_classes, num_transformer_layers=args.encoders, mlp_size=args.mlp_size, patch_size=args.patch_size, embedding_dim=args.emb_dim, num_heads=args.num_heads).to(device)
    try:
        model.load_state_dict(torch.load(model_weights_path)['model'])
    except FileNotFoundError:
        print("Model not found")
        model.load_state_dict(torch.load(f'logFile/{args.log_name}/{arch}/last_{args.fold_num}.pt')['model'])
    model.eval()

    # Assuming the number of classes is 5

    total_samples, correct_predictions = 0, 0
    CM = np.zeros((n_classes, n_classes), dtype=int)
    assert args.test_batch_size==1, "Test batch size is not 1, the confusion matrix implementation will not work for it"
    with torch.no_grad():
        for inputs, labels, _ in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Update counters for each class
            for c in range(n_classes):
                predicted_class = (predicted == c)
                true_class = (labels == c)
                if predicted_class or true_class:
                    if predicted_class and true_class:
                        CM[c,c]+=1
                        break
                    elif predicted_class:
                        CM[c,labels[0]]+=1
                        break
                    elif true_class:
                        CM[predicted[0], c]+=1
                        break
                    else:
                        print("Something wrong in confusion matrix")
                        break

    with open(args.save_path, 'a', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(["TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall"])
        # writer.writerow([true_positives, false_positives, true_negatives, false_negatives, accuracy, precision, recall])
        writer.writerows(CM)
        writer.writerow([])

def plot_wrong(model, test_dataloader, weights, img_save_path=None, fold=1, arch="trans_pre"):
    if img_save_path is None:
        img_save_path = args.img_save_path
    model.to(device)
    che = torch.load(weights)
    model.load_state_dict(che['model'])
    model.eval()
    wrong_array = None
    wrong_name = None
    for x_test, y_test, name_test in test_dataloader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        with torch.no_grad():
            outputs = model(x_test)
            _, predicted = torch.max(outputs.data, 1)                        
            
            if not (predicted == y_test).sum().item()==1:               
                if wrong_array is None:
                    wrong_array = x_test.cpu().detach().numpy().reshape(-1)
                    wrong_name = np.array([str(name_test[0])+args.data_names[predicted[0]]])
                else:
                    wrong_array = np.vstack((wrong_array, x_test.cpu().detach().numpy().reshape(-1)))
                    wrong_name = np.concatenate((wrong_name, np.array([str(name_test[0])+args.data_names[predicted[0]]])))
    if wrong_array is not None:
        print(wrong_array.shape)
        if len(wrong_array.shape)==1:
            wrong_array = wrong_array.reshape(1, -1)
        plot9Data(wrong_array, img_save_path+arch+str(fold)+"_", all=True, name=wrong_name)


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    # parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--weights', type=str, default='last_', help='.pt file')

    parser.add_argument('--log_name', type=str, default='Final', help='Read and Save log file name')
    parser.add_argument('--data_dir', type=str, default='data/data_war_sngp/', help='input data directoy containing .npy files')
    parser.add_argument('--baseline', action='store_true', help='Is it a baseline corected data.')
    parser.add_argument('--arg_file', type=str, default='logFile/NoName/FC/arguments_1.txt', help='input argument .txt file')

    parser.add_argument('--model', default="incep", type=str, help='model')
    parser.add_argument('--FC', action='store_true', help='Set the dataset format to (n, l). Where n is number of sample and l is length.')
    parser.add_argument('--weight_folder', default="", type=str, help='model')
# Hyp
    
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--plot', default=True, type=bool, help='plot fake and real dataa examples if true')
    parser.add_argument('--plot_wrong', default=True, type=bool, help='plot data when predicted wrong if true')
    parser.add_argument('--save_path', type=str, default='default', help='.csv file to save CM')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--test_batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--check_epoch', default=10, type=int, help='evaluate performance after epochs')


# Transformer Hyp
    parser.add_argument('--encoders', default=3, type=int, help='Number of transformer encoder')
    parser.add_argument('--mlp_size', default=64, type=int, help='Feed forword size')
    parser.add_argument('--num_heads', default=4, type=int, help='Number of transformer encoder head')
    parser.add_argument('--patch_size', default=20, type=int, help='Embending patch size')
    parser.add_argument('--input_size', default=4000, type=int, help='Embending patch size')
    parser.add_argument('--emb_dim', default=20, type=int, help='Dimenstion of input embeding')
    

    args = parser.parse_args()
    m = args.model
    weights = args.weights
    load_args_from_txt(args.arg_file, args)
    args.weights = weights
    if args.baseline:
        args.save_path = f"logFile/{args.log_name}/test/{args.weights}{args.model}_baseline.csv"
    else:
        args.save_path = f"logFile/{args.log_name}/test/{args.weights}{args.model}.csv"

    print(args.data_dir)
    if not os.path.exists(os.path.join("logFile",args.log_name, "test")):
        os.mkdir(os.path.join("logFile",args.log_name, "test"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eddie_data = False

    options = vars(args)
    print(options)
    if args.model=="all":
        pass
    elif args.model=="trans_avg":
        print("Using SpectroscopyTransformerEncoder_PreT model")
        model = SpectroscopyTransformerEncoder_PreT
    elif args.model=="incep":
        print("Using InceptionNetwork model")
        model = InceptionNetwork
    elif args.model=="incep_avg":
        print("Using InceptionNetwork_PreT model")
        model = InceptionNetwork_PreT
    elif args.model=="trans":
        print("Using SpectroscopyTransformerEncoder model")
        model = SpectroscopyTransformerEncoder
    elif args.model=="FC":
        print("Using SpectroscopyTransformerEncoder model")
        assert args.FC, f"FC modell required argument args.FC to be True"
        model = FCNet
    elif args.model=="res":
        print("Using SpectroscopyTransformerEncoder model")
        model = PSDNResNet
    else:
        raise ValueError("No model selected")



    for fold_name in os.listdir(f'logFile/{args.log_name}/data_split'):
        args.fold_num = int(fold_name)
        folder_path = os.path.join(f'logFile/{args.log_name}/data_split', fold_name)
        
        weight_path = args.model
        if args.baseline:
            weight_path = weight_path + "_baseline"

        weight_path = os.path.join("logFile", args.log_name, weight_path, args.weights + str(args.fold_num) + ".pt")
        
        if os.path.exists(os.path.join(folder_path, "index_test.txt")):
            dataarrayX_test, dataarrayY_test, data_number_test, data_list = dataset_from_txt(args, os.path.join(folder_path, "index_test.txt"))
            args.data_names = data_list
            dataset_test = IRDatasetFromNames(dataarrayX_test, dataarrayY_test, data_number_test, FC=args.FC)
            test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=True)
            print(f"Test data size {len(dataset_test)}")
            test(args, model, test_dataloader, weight_path, n_classes=len(np.unique(dataarrayY_test)), arch=args.model)
        
        else:
            dataarrayX_val, dataarrayY_val, data_number_val, data_list = dataset_from_txt(args, os.path.join(folder_path, "index_val.txt"))
            args.data_names = data_list

            dataset_val = IRDatasetFromNames(dataarrayX_val, dataarrayY_val, data_number_val, FC=args.FC)    
            val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
            print(f"Validation data size {len(dataset_val)}")

            test(args, model, val_dataloader, weight_path, n_classes=len(np.unique(dataarrayY_val)), arch=args.model)
