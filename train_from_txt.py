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
from model import SpectroscopyTransformerEncoder, SpectroscopyTransformerEncoder_PreT, InceptionNetwork, InceptionNetwork_PreT
from utils import plot9Data, save_arguments_to_file, load_args_from_txt
from eval import test

import os
import csv

def train(args, model, train_dataloader,val_dataloader, n_classes=2, device='cpu', arch="trans_pre"):
    if arch=="incep" or arch=="incep_pre":
        model = model(n_classes)
    else:
        model = model(num_classes=n_classes, num_transformer_layers=args.encoders, mlp_size=args.mlp_size, patch_size=args.patch_size, embedding_dim=args.emb_dim, num_heads=args.num_heads)
    # model = Net()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),
                            lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # for parameter in model.parameters():
    #     if parameter.dim()>1:
    #         nn.init.xavier_uniform_(parameter)
    best_test_accuracy = 0

    for epoch in range(args.epochs):
        model.train()
        for x_train, y_train, _ in train_dataloader:
            y_train = y_train.type(torch.LongTensor)
            x_train = x_train.to(device)
            # if len(x_train<batch_size):
            #     continue
            y_train = y_train.to(device)
            outputs = model(x_train)
            loss = loss_fn(outputs, y_train)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0 and val_dataloader is not None:
            model.eval()
            test_correct = 0
            test_size = 0
            for x_test, y_test, _ in val_dataloader:
                test_size+=len(y_test)
                x_test = x_test.to(device)
                y_test = y_test.type(torch.LongTensor)
                y_test = y_test.to(device)
                with torch.no_grad():
                    outputs = model(x_test)
                    _, predicted = torch.max(outputs.data, 1)                        
                    test_correct += (predicted == y_test).sum().item()                
            test_accuracy = test_correct/test_size                        
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Test accuracy = {test_accuracy:.4f}")
            if best_test_accuracy<=test_accuracy:
                best_test_accuracy = test_accuracy
                best_epoch = epoch+1
                print(f'saving weight for epoch {epoch+1}')
                torch.save(
                            {
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                            },
                            f'logFile/{args.log_name}/{args.weight_folder}/best_{args.fold_num}.pt',
                )
    print(f'saving weight for last')
    torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f'logFile/{args.log_name}/{args.weight_folder}/last_{args.fold_num}.pt',
    )
    if val_dataloader:
        with open(os.path.join("logFile",args.log_name, f'Val_results_{args.weight_folder}.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["best_epoch", "Max_Correct"])
            writer.writerow([best_epoch, best_test_accuracy])
        if args.plot:
            if args.plot_wrong:
                assert args.test_batch_size==1, f"Ask for plot worng prediction but batch size is {args.test_batch_size}"
                plot_wrong(args, model, val_dataloader, f'logFile/{args.log_name}/{args.weight_folder}/best_{args.fold_num}.pt', f'logFile/{args.log_name}/', fold=args.fold_num)


def plot_wrong(args, model, test_dataloader, weights, img_save_path=None, fold=1):
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
        plot9Data(wrong_array, img_save_path+args.weight_folder+str(fold)+"_", all=True, name=wrong_name)

def prepare_folders(args):
    if args.baseline:
        args.weight_folder = args.model+"_baseline"
    else:
        args.weight_folder = args.model

    args.save_path = f'logFile/{args.log_name}/results_{args.weight_folder}.csv'

    os.mkdir(f'logFile/{args.log_name}/{args.weight_folder}')
    return args


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    # parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--log_name', type=str, default='check', help='Save log file name')
    parser.add_argument('--data_dir', type=str, default='data/data_warwick/', help='input data directoy containing .npy files')
    # parser.add_argument('--baseline', default=True, help='Is it a baseline corected data.')
    parser.add_argument('--baseline', action='store_true', help='Is it a baseline corected data.')
    parser.add_argument('--FC', action='store_true', help='Set the dataset format to (n, l). Where n is number of sample and l is length.')
    parser.add_argument('--weighted_loss', action='store_true', help='Set the weighted loss to true.')


# Hyp

    parser.add_argument('--model', default="trans_pre", type=str, help='model')
    parser.add_argument('--weight_folder', default="default", type=str, help='model')


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



    n = args.baseline
    m = args.model
    load_args_from_txt(f'logFile/{args.log_name}/arguments.txt', args)
    args.model = m
    args.baseline = n

    args = prepare_folders(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model=="all":
        pass
    elif args.model=="trans_pre":
        print("Using SpectroscopyTransformerEncoder_PreT model")
        model = SpectroscopyTransformerEncoder_PreT
    elif args.model=="incep":
        print("Using InceptionNetwork model")
        model = InceptionNetwork
    elif args.model=="incep_pre":
        print("Using InceptionNetwork_PreT model")
        model = InceptionNetwork_PreT
    elif args.model=="trans":
        print("Using SpectroscopyTransformerEncoder model")
        model = SpectroscopyTransformerEncoder
    else:
        raise ValueError("No model selected")
    exp_num = 1   
    while exp_num<1000:
        if os.path.exists(f'logFile/{args.log_name}/{args.weight_folder}/arguments_{exp_num}.txt'):
            exp_num+=1
            continue
        else:
            save_arguments_to_file(args, f'logFile/{args.log_name}/{args.weight_folder}/arguments_{exp_num}.txt')
            break


    # for fold_name in os.listdir(f'logFile/{args.log_name}/data_split'):
    for i in range(10):
        fold_name = str(i)
        args.fold_num = int(fold_name)
        folder_path = os.path.join(f'logFile/{args.log_name}/data_split', fold_name)
        dataarrayX_train, dataarrayY_train, data_number_train, data_list = dataset_from_txt(args, os.path.join(folder_path, "index_train.txt"))
        dataarrayX_val, dataarrayY_val, data_number_val, _ = dataset_from_txt(args, os.path.join(folder_path, "index_val.txt"))
        args.data_names = data_list

        dataset_train = IRDatasetFromNames(dataarrayX_train, dataarrayY_train, data_number_train, FC=args.FC)
        dataset_val = IRDatasetFromNames(dataarrayX_val, dataarrayY_val, data_number_val, FC=args.FC)    
        val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print(f"Training data size {len(dataset_train)}")
        print(f"Validation data size {len(dataset_val)}")

        if os.path.exists(os.path.join(folder_path, "index_test.txt")):
            dataarrayX_test, dataarrayY_test, data_number_test, _ = dataset_from_txt(args, os.path.join(folder_path, "index_test.txt"))
            dataset_test = IRDatasetFromNames(dataarrayX_test, dataarrayY_test, data_number_test, FC=args.FC)
            test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=True)
            print(f"Test data size {len(dataset_test)}")
            train(args, model, train_dataloader,val_dataloader, n_classes=len(np.unique(dataarrayY_train)), device=device, arch=args.model)
            test(args, model, test_dataloader, f'logFile/{args.log_name}/{args.weight_folder}/best_{args.fold_num}.pt', n_classes=len(np.unique(dataarrayY_train)), arch=args.model)
        else:
            train(args, model, train_dataloader,val_dataloader, n_classes=len(np.unique(dataarrayY_train)), device=device, arch=args.model)
