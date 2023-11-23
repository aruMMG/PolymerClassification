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

def train(args, model, train_dataloader,val_dataloader, n_classes=2, device='cpu', arch="pre"):
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
                            f'logFile/{args.log_name}/weights_{arch}/best_{args.fold_num}.pt',
                )
    print(f'saving weight for last')
    torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f'logFile/{args.log_name}/weights_{arch}/last_{args.fold_num}.pt',
    )
    if val_dataloader:
        with open(os.path.join("logFile",args.log_name, f'results_{arch}.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["best_epoch", "Max_Correct"])
            writer.writerow([best_epoch, best_test_accuracy])
        if args.plot:
            if args.plot_wrong:
                assert args.test_batch_size==1, f"Ask for plot worng prediction but batch size is {args.test_batch_size}"
                plot_wrong(model, val_dataloader, f'logFile/{args.log_name}/weights_{arch}/best_{args.fold_num}.pt', name_tuple, f'logFile/{args.log_name}/', fold=args.fold_num, arch=arch)

def test(model_class, test_dataloader, model_weights_path, n_classes=2, arch="pre"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create an instance of the model
    if arch=="incep" or arch=="incep_pre":
        model = model_class(n_classes).to(device)
    else:
        model = model_class(num_classes=n_classes, num_transformer_layers=args.encoders, mlp_size=args.mlp_size, patch_size=args.patch_size, embedding_dim=args.emb_dim, num_heads=args.num_heads).to(device)
    try:
        model.load_state_dict(torch.load(model_weights_path)['model'])
    except FileNotFoundError:
        model.load_state_dict(torch.load(f'logFile/{args.log_name}/weights_{arch}/last_{args.fold_num}.pt')['model'])
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

    with open(os.path.join("logFile",args.log_name, f'test_results_{arch}.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(["TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall"])
        # writer.writerow([true_positives, false_positives, true_negatives, false_negatives, accuracy, precision, recall])
        writer.writerows(CM)
        writer.writerow([])

def plot_wrong(model, test_dataloader, weights, name_tuple, img_save_path=None, fold=1, arch="pre"):
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
                    wrong_name = np.array([name_tuple[int(name_test[0])]])
                else:
                    wrong_array = np.vstack((wrong_array, x_test.cpu().detach().numpy().reshape(-1)))
                    wrong_name = np.concatenate((wrong_name, np.array([name_tuple[int(name_test[0])]])))
    if wrong_array is not None:
        print(wrong_array.shape)
        if len(wrong_array.shape)==1:
            wrong_array = wrong_array.reshape(1, -1)
        plot9Data(wrong_array, img_save_path+arch+str(fold)+"_", all=True, name=wrong_name)


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    # parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--log_name', type=str, default='TransformerExp', help='Save log file name')
    parser.add_argument('--img_save_path', type=str, default='plots/wrong/HDPE_LDPE', help='Save wrong predicted images')

# Hyp
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--encoders', default=3, type=int, help='Number of transformer encoder')
    parser.add_argument('--mlp_size', default=64, type=int, help='Feed forword size')
    parser.add_argument('--num_heads', default=4, type=int, help='Number of transformer encoder head')
    parser.add_argument('--patch_size', default=20, type=int, help='Embending patch size')
    parser.add_argument('--emb_dim', default=20, type=int, help='Dimenstion of input embeding')
    parser.add_argument('--model', default="pre", type=str, help='model')
    
    
    parser.add_argument('--train_perc', default=0.8, type=float, help='percent used as training dataset')
    parser.add_argument('--val_perc', default=1, type=float, help='percent used as training dataset')
    parser.add_argument('--test_perc', default=0, type=float, help='percent used as training dataset')
    parser.add_argument('--folds', default=2, type=int, help='number of epochs')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--test_batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--check_epoch', default=10, type=int, help='evaluate performance after epochs')
    parser.add_argument('--plot', default=True, type=bool, help='plot fake and real dataa examples if true')
    parser.add_argument('--plot_wrong', default=True, type=bool, help='plot data when predicted wrong if true')

    args = parser.parse_args()
    log_name = createFolders(args)
    args.log_name = log_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eddie_data = False
    if eddie_data:
        name_HDPE = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/HDPE_name.npy")
        name_LDPE = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/LDPE_name.npy")
        name_PET = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/PET_name.npy")
        name_PP = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/PP_name.npy")
        # name_others = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/others_name.npy")
        

        hdpe_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/HDPE.npy")
        ldpe_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/LDPE.npy")
        pet_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/PET.npy")
        pp_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/PP.npy")
        # other_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/others.npy")

    else:
        print("Running on Open Specy")
        # name_all = np.load("data/real/FTIR_name.npy")
        # name_notPolymer = np.array([i for i in name_all if i.startswith("notPolymer")])
        # name_HDPE = np.array([i for i in name_all if i.startswith("HDPE")])
        # name_LDPE = np.array([i for i in name_all if i.startswith("LDPE")])
        # name_PET = np.array([i for i in name_all if i.startswith("PET")])
        # name_PP = np.array([i for i in name_all if i.startswith("PP")])
        name_HDPE = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/HDPE_name.npy")
        name_LDPE = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/LDPE_name.npy")
        name_PET = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/PET_name.npy")
        # name_PP = np.load("data/open_specy/from_eddie_raw/PP_name.npy")
        # name_others = np.load("data/open_specy/from_eddie_raw/others_name.npy")

        hdpe_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/HDPE_baseline.npy") 
        ldpe_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/LDPE_baseline.npy") 
        pet_array = np.load("data/Data_Edward/Data/FTIR/data_Eddie/exp2/PET_baseline.npy") 
        # pp_array = np.load("data/open_specy/from_eddie_raw/PP_baseline.npy") 
        # other_array = np.load("data/open_specy/from_eddie_raw/others_baseline.npy") 
    hdpe_Y = np.ones(hdpe_array.shape[0], dtype=float)*0
    ldpe_Y = np.ones(ldpe_array.shape[0], dtype=float)*0
    X = (hdpe_array, ldpe_array, pet_array)
    name_tuple = np.concatenate((name_HDPE, name_LDPE, name_PET))
    if args.plot_wrong:
        assert len(hdpe_array)==len(name_HDPE), f"name of sample not available"
        assert len(ldpe_array)==len(name_LDPE), f"name of sample not available"

    dataset = IRDatasetFromArray(X)
    if args.model=="all":
        pass
    elif args.model=="pre":
        print("Using SpectroscopyTransformerEncoder_PreT model")
        model = SpectroscopyTransformerEncoder_PreT
    elif args.model=="incep":
        print("Using InceptionNetwork model")
        model = InceptionNetwork
    elif args.model=="incep_pre":
        print("Using InceptionNetwork_PreT model")
        model = InceptionNetwork_PreT
    else:
        print("Using SpectroscopyTransformerEncoder model")
        model = SpectroscopyTransformerEncoder
    save_arguments_to_file(args, f'logFile/{args.log_name}/arguments.txt')
    for fold in range(args.folds):
        args.fold_num = fold
        train_size = int(args.train_perc * len(dataset))
        if args.val_perc!=1.0:
            val_size = int(args.val_perc * len(dataset))
            test_size = len(dataset) - train_size - val_size
            train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
            print(f"Training data size {train_size}")
            print(f"Test data size {test_size}")
            print(f"Validation data size {val_size}")
        else:
        # train_size = int(args.train_perc * len(train_dataset1))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
            val_dataloader = None
            print(f"Training data size {train_size}")
            print(f"Test data size {test_size}")
        if args.model=="all":
            print("Using SpectroscopyTransformerEncoder_PreT model")
            model = SpectroscopyTransformerEncoder_PreT
            if args.test_perc==1.0:
                train(args, model, train_dataloader,val_dataloader, n_classes=len(X), device=device, arch="pre")
                test(model, test_dataloader, f'logFile/{args.log_name}/weights_pre/best_{args.fold_num}.pt', n_classes=len(X), arch="pre")
            elif val_dataloader is None:
                train(args, model, train_dataloader,test_dataloader, n_classes=len(X), device=device, arch="pre")
            print("Using InceptionNetwork model")
            model = InceptionNetwork
            if args.test_perc==1.0:
                train(args, model, train_dataloader,val_dataloader, n_classes=len(X), device=device, arch="incep")
                test(model, test_dataloader, f'logFile/{args.log_name}/weights_incep/best_{args.fold_num}.pt', n_classes=len(X), arch="incep")
            elif val_dataloader is None:
                train(args, model, train_dataloader,test_dataloader, n_classes=len(X), device=device, arch="incep")
            print("Using InceptionNetwork_PreT model")
            model = InceptionNetwork_PreT
            if args.test_perc==1.0:
                train(args, model, train_dataloader,val_dataloader, n_classes=len(X), device=device, arch="incep_pre")
                test(model, test_dataloader, f'logFile/{args.log_name}/weights_incep_pre/best_{args.fold_num}.pt', n_classes=len(X), arch="incep_pre")
            elif val_dataloader is None:
                train(args, model, train_dataloader,test_dataloader, n_classes=len(X), device=device, arch="incep_pre")

            print("Using SpectroscopyTransformerEncoder model")
            model = SpectroscopyTransformerEncoder
            if args.test_perc==1.0:
                train(args, model, train_dataloader,val_dataloader, n_classes=len(X), device=device, arch="trans")
                test(model, test_dataloader, f'logFile/{args.log_name}/weights_trans/best_{args.fold_num}.pt', n_classes=len(X), arch="trans")
            elif val_dataloader is None:
                train(args, model, train_dataloader,test_dataloader, n_classes=len(X), device=device, arch="trans")
        else:
            if args.test_perc==1.0:
                train(args, model, train_dataloader,val_dataloader, n_classes=len(X), device=device, arch=args.model)
                test(model, test_dataloader, f'logFile/{args.log_name}/weights_{args.model}/best_{args.fold_num}.pt', n_classes=len(X), arch=args.model)
            elif val_dataloader is None:
                train(args, model, train_dataloader,test_dataloader, n_classes=len(X), device=device, arch="pre")
