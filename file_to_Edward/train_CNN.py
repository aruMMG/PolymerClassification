import torch
from model_inc import Net_Inception
from model import FCNet, Net
from dataset import IRDataset, IRDatasetFromArray, IRDatasetFromArrayFC
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from utils import createFolders, plot9Data

import os
import csv

def trainPlain(model, dataloader, criteriaon, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    for data, labels in dataloader:
        data = data.to(device)
        data = data.to(dtype=torch.float32)
        labels = labels.to(device)
        labels = labels.unsqueeze(1).to(dtype=torch.float32)
        output = model(data)
        loss = criteriaon(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad(): 
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for data, labels in dataloader:
            data = data.to(device)
            data = data.to(dtype=torch.float32)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)
            
            output = model(data)
            loss = criterion(output, labels)
            labels = labels.squeeze()

            total_loss += loss.item() * data.size(0)

            _, predictions = output.max(dim=1)

            total_correct += (predictions == labels).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
def train(Model, dataset, n_classes=2, device='cpu', name_tuple=None):
    train_size = int(args.train_perc * len(dataset))
    test_size = len(dataset) - train_size
    
    for folds in range(args.folds):
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
        model = Model(num_class=n_classes)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for parameter in model.parameters():
            if parameter.dim()>1:
                nn.init.xavier_uniform_(parameter)
        
        best_test_accuracy = 0
        for epoch in range(args.epochs):
            model.train()
            for x_train, y_train, _ in train_dataloader:
                x_train = x_train.to(device)
                # if len(x_train<batch_size):
                #     continue
                y_train = y_train.to(device)
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                model.eval()
                test_correct = 0
                for x_test, y_test, _ in test_dataloader:
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    with torch.no_grad():
                        outputs = model(x_test)
                        _, predicted = torch.max(outputs.data, 1)                        
                        test_correct += (predicted == y_test).sum().item()                
                        
                test_accuracy = test_correct/test_size
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Test accuracy = {test_correct/test_size:.4f}")
                
                if best_test_accuracy<test_accuracy:
                    best_test_accuracy=test_accuracy
                    best_epoch = epoch+1
                    print(f'saving weight for epoch {epoch+1}')
                    torch.save(
                                {
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                },
                                f'logFile/{args.log_name}/weights/best_{folds}.pt',
                                )
        with open(os.path.join("logFile",args.log_name, 'results.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["best_epoch", "Max_Correct"])
            writer.writerow([best_epoch, best_test_accuracy])
        if args.plot:
            if args.plot_wrong:
                assert args.test_batch_size==1, f"Ask for plot worng prediction but batch size is {args.test_batch_size}"
                plot_wrong(Model, test_dataloader, n_classes, f'logFile/{args.log_name}/weights/best_{folds}.pt', name_tuple, fold=folds)

def plot_wrong(Model, test_dataloader, n_classes, weights, name_tuple, fold=1):
    model = Model(num_class=n_classes)
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
            current_wrong_array = x_test.cpu().detach().numpy()
            if len(current_wrong_array.shape)>1:
                current_wrong_array = current_wrong_array.reshape(-1)
            if not (predicted == y_test).sum().item()==1:             
                if wrong_array is None:
                    wrong_array = current_wrong_array
                    wrong_name = np.array([name_tuple[int(name_test[0])]])
                else:
                    wrong_array = np.vstack((wrong_array, current_wrong_array))
                    wrong_name = np.concatenate((wrong_name, np.array([name_tuple[int(name_test[0])]])))
    if wrong_array is not None:
        plot9Data(wrong_array, args.img_save_path+str(fold), all=True, name=wrong_name)



if __name__=="__main__":
    # dataset = IRDataset("data")
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    # parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--log_name', type=str, default='Exp', help='data type ex: FTIR, Ramen, LIBS')
    parser.add_argument('--img_save_path', type=str, default='plots/wrong/HDPE_LDPE', help='Save wrong predicted images')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--train_perc', default=0.8, type=float, help='percent used as training dataset')
    parser.add_argument('--folds', default=1, type=int, help='number of epochs')
    parser.add_argument('--epochs', default=21, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--test_batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--check_epoch', default=10, type=int, help='evaluate performance after epochs')
    parser.add_argument('--plot', default=False, type=bool, help='plot fake and real dataa examples if true')
    parser.add_argument('--plot_wrong', default=False, type=bool, help='plot fake and real dataa examples if true')

    args = parser.parse_args()
    log_name = createFolders(args)
    args.log_name = log_name
    print(args.plot)
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


    assert len(hdpe_array)==len(name_HDPE), "HDPE array length not match name length"
    assert len(ldpe_array)==len(name_LDPE), "LDPE array length not match name length"
    dataset_tuple = (hdpe_array, ldpe_array, pet_array, pp_array)
    name_tuple = np.concatenate((name_HDPE, name_LDPE, name_PET, name_PP))
    
    dataset = IRDatasetFromArray(dataset_tuple)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(dataset_tuple)
    train(Net, dataset, n_classes, device=device, name_tuple=name_tuple)
    # for fold in range(10):
    #     train_size = int(0.8 * len(dataset))
    #     test_size = len(dataset) - train_size
    #     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    #     test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    #     model = Net_Inception()

    #     model = model.to(device)
        
    #     for parameter in model.parameters():
    #         if parameter.dim()>1:
    #             nn.init.xavier_uniform_(parameter)

    #     optimizer = optim.Adam(model.parameters(), lr=0.001)
    #     criterion = nn.MSELoss()

    #     num_epochs = 100
    #     for epoch in range(num_epochs):
    #         train(model, train_dataloader, criterion, optimizer)
        
    #     loss, acc = test(model, test_dataloader, criterion)
    #     print(f"fold result {loss}, {acc}")