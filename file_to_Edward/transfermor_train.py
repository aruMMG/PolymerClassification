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
from model import SpectroscopyTransformerEncoder, SpectroscopyTransformerEncoder_PreT
from utils import createFolders, plot9Data

import os
import csv

def train(args, model, train_dataloader,test_dataloader, n_classes=2, device='cpu'):
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
        
        if epoch % 10 == 0:
            model.eval()
            test_correct = 0
            test_size = 0
            for x_test, y_test, _ in test_dataloader:
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
                            f'logFile/{args.log_name}/weights/best_{args.fold_num}.pt',
                )
    with open(os.path.join("logFile",args.log_name, 'results.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["best_epoch", "Max_Correct"])
        writer.writerow([best_epoch, best_test_accuracy])
    if args.plot:
        if args.plot_wrong:
            assert args.test_batch_size==1, f"Ask for plot worng prediction but batch size is {args.test_batch_size}"
            plot_wrong(model, test_dataloader, f'logFile/{args.log_name}/weights/best_{args.fold_num}.pt', name_tuple, fold=args.fold_num)

def plot_wrong(model, test_dataloader, weights, name_tuple, fold=1):
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
                    wrong_name = np.array([name_tuple[int(name_test[0])]])
                else:
                    wrong_array = np.vstack((wrong_array, x_test.cpu().detach().numpy().reshape(-1)))
                    wrong_name = np.concatenate((wrong_name, np.array([name_tuple[int(name_test[0])]])))
    if wrong_array is not None:
        print(wrong_array.shape)
        plot9Data(wrong_array, args.img_save_path+str(fold), all=True, name=wrong_name)


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    # parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--log_name', type=str, default='TransformerExp', help='Save log file name')
    parser.add_argument('--img_save_path', type=str, default='plots/wrong/HDPE_LDPE', help='Save wrong predicted images')

# Hyp
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--encoders', default=3, type=int, help='Number of transformer encoder')
    parser.add_argument('--mlp_size', default=256, type=int, help='Feed forword size')
    parser.add_argument('--num_heads', default=4, type=int, help='Number of transformer encoder head')
    parser.add_argument('--patch_size', default=20, type=int, help='Embending patch size')
    parser.add_argument('--emb_dim', default=20, type=int, help='Dimenstion of input embeding')
    
    
    parser.add_argument('--train_perc', default=0.8, type=float, help='percent used as training dataset')
    parser.add_argument('--folds', default=2, type=int, help='number of epochs')
    parser.add_argument('--epochs', default=51, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--test_batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--check_epoch', default=10, type=int, help='evaluate performance after epochs')
    parser.add_argument('--plot', default=False, type=bool, help='plot fake and real dataa examples if true')
    parser.add_argument('--plot_wrong', default=False, type=bool, help='plot data when predicted wrong if true')

    args = parser.parse_args()
    log_name = createFolders(args)
    args.log_name = log_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eddie_data = True
    if eddie_data:
        name_HDPE = np.load("data/Data_Edward/Data/FTIR/data_Eddie/HDPE_name.npy")
        name_LDPE = np.load("data/Data_Edward/Data/FTIR/data_Eddie/LDPE_name.npy")
        name_PET = np.load("data/Data_Edward/Data/FTIR/data_Eddie/PET_name.npy")
        name_PP = np.load("data/Data_Edward/Data/FTIR/data_Eddie/PP_name.npy")

        hdpe_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/HDPE.npy")
        ldpe_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/LDPE.npy")
        pet_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/PET.npy")
        pp_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/PP.npy")
    else:
        name_all = np.load("data/real/FTIR_name.npy")
        name_notPolymer = np.array([i for i in name_all if i.startswith("notPolymer")])
        name_HDPE = np.array([i for i in name_all if i.startswith("HDPE")])
        name_LDPE = np.array([i for i in name_all if i.startswith("LDPE")])
        name_PET = np.array([i for i in name_all if i.startswith("PET")])
        name_PP = np.array([i for i in name_all if i.startswith("PP")])
        hdpe_array = np.load("data/real/HDPE_norm_no_neg_data.npy")
        ldpe_array = np.load("data/real/LDPE_norm_no_neg_data.npy")
        pet_array = np.load("data/real/PET_norm_no_neg_data.npy")
        pp_array = np.load("data/real/PP_norm_no_neg_data.npy")
        notPolymer_array = np.load("data/real/other_norm_no_neg_data.npy")

    hdpe_Y = np.ones(hdpe_array.shape[0], dtype=float)*0
    ldpe_Y = np.ones(ldpe_array.shape[0], dtype=float)*0
    X = (hdpe_array, ldpe_array, pet_array, pp_array)
    name_tuple = np.concatenate((name_HDPE, name_LDPE, name_PET, name_PP))


    dataset = IRDatasetFromArray(X)
    

    for fold in range(args.folds):
        args.fold_num = fold
        train_size = int(args.train_perc * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
        train(args, SpectroscopyTransformerEncoder_PreT, train_dataloader,test_dataloader, n_classes=len(X), device=device)