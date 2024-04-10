import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import IRDatasetFromNames, dataset_from_txt

def target_category_loss(x, category_index, nb_classes):
    return torch.mul(x, F.one_hot(category_index, nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        # print(output.size())
        return output[target_category]

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            output = output.squeeze()
            target_category = np.argmax(output.cpu().data.numpy())
            # print(output)
            # print(target_category)
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        #weights = np.mean(grads, axis=(0))
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
             cam += w * activations[i, :]
        # cam = activations.T.dot(weights)
        # cam = activations.dot(weights)
        # cam = activations.dot(weights)
        # print(input_tensor.shape[1])
        # print(cam.shape)
        # x = np.arange(0, 247, 1)
        # plt.plot(x, cam.reshape(-1, 1))
        # sns.set()
        # ax = sns.heatmap(cam.reshape(-1, 1).T)
        #cam = cv2.resize(cam, input_tensor.shape[1:][::-1])
        #cam = resize_1d(cam, (input_tensor.shape[2]))
        cam = np.interp(np.linspace(0, cam.shape[0], input_tensor.shape[2]), np.linspace(0, cam.shape[0], cam.shape[0]), cam)   #Change it to the interpolation algorithm that numpy comes with.
        #cam = np.maximum(cam, 0)
        # cam = np.expand_dims(cam, axis=1)
        # ax = sns.heatmap(cam)
        # plt.show()
        # cam = cam - np.min(cam)
        # cam = cam / np.max(cam)
        heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)#归一化处理
        # heatmap = (cam - np.mean(cam, axis=-1)) / (np.std(cam, axis=-1) + 1e-10)
        print(heatmap.shape)
        return heatmap
class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(activations, axis=1)
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None] * grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=1)
        return weights
# from pytorch_grad_cam.utils.image import preprocess_image
import matplotlib.pyplot as plt
def plot_and_save(data1, data2, filename='plot.png'):
    """
    Plots two one-dimensional data series on the same plot and saves it as an image.

    Parameters:
        data1 (list or numpy array): First one-dimensional data series.
        data2 (list or numpy array): Second one-dimensional data series.
        filename (str): Filename to save the plot image. Default is 'plot.png'.
    """
    # Create a new figure
    fig, ax1 = plt.subplots()

    # Plot data1
    ax1.plot(data1, label='Data 1', color='tab:blue')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Data 1', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for data2
    ax2 = ax1.twinx()
    ax2.plot(data2, label='Data 2', color='tab:red')
    ax2.set_ylabel('Data 2', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Save the plot as an image
    plt.savefig(filename)

    # Close the plot to release resources
    plt.close()





if __name__=="__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    parser.add_argument('--log_name', type=str, default='EXP6', help='Read and Save log file name')
    parser.add_argument('--save_path', type=str, default='./logFile/EXP6/c/test/plots/', help='input data directoy containing .npy files')
    parser.add_argument('--data_dir', type=str, default='data/data_warwick/', help='input data directoy containing .npy files')
    parser.add_argument('--fold_name', type=str, default='0', help='input data directoy containing .npy files')
    parser.add_argument('--weight', type=str, default="./logFile/EXP6/c/incep/last_0.pt", help='input data directoy containing .npy files')
    parser.add_argument('--baseline', action='store_true', help='Is it a baseline corected data.')
    parser.add_argument('--FC', action='store_true', help='Set the dataset format to (n, l). Where n is number of sample and l is length.')
    args = parser.parse_args()
    
    from model import InceptionNetwork, SpectroscopyTransformerEncoder_PreT, InceptionNetwork_PreT, SpectroscopyTransformerEncoder
    # from model import SpectroscopyTransformerEncoder_PreT, InceptionNetwork_PreT
    
    model = InceptionNetwork(4)
    model.load_state_dict(torch.load(args.weight)['model'])
    target_layer = model.inception1

    # model = InceptionNetwork_PreT(4)
    # model.load_state_dict(torch.load(args.weight)['model'])
    # target_layer = model.IR_PreT.inception1

    # model = SpectroscopyTransformerEncoder_PreT(num_classes=4, mlp_size=64)
    # model.load_state_dict(torch.load(args.weight)['model'])
    # target_layer = model.IR_PreT.transformer_encoder
    
    # model = SpectroscopyTransformerEncoder(num_classes=4, mlp_size=64)
    # model.load_state_dict(torch.load(args.weight)['model'])
    # target_layer = model.transformer_encoder
    
    
    net = GradCAM(model, target_layer)
  




    # args.baseline = True
    # for fold_name in os.listdir(f'logFile/{args.log_name}/data_split'):
    #     args.fold_num = int(fold_name)
    folder_path = os.path.join(f'logFile/{args.log_name}/data_split', args.fold_name)

    dataarrayX_val, dataarrayY_val, data_number_val, data_list = dataset_from_txt(args, os.path.join(folder_path, "index_val.txt"))
    data_names = data_list

    dataset_val = IRDatasetFromNames(dataarrayX_val, dataarrayY_val, data_number_val, FC=args.FC)    
    val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, drop_last=True)
    print(f"Validation data size {len(dataset_val)}")

    # test(args, model, val_dataloader, weight_path, n_classes=len(np.unique(dataarrayY_val)), arch=args.model)

    heatmap0, heatmap1, heatmap2, heatmap3 = None, None, None, None
    input_tensors0, input_tensors1, input_tensors2, input_tensors3 = None, None, None, None
    for input_tensor, labels, _ in val_dataloader:
        # inputs, labels = inputs.to(device), labels.to(device)
        # outputs = model(inputs)
        # _, predicted = torch.max(outputs.data, 1)
        output = net(input_tensor)
        input_tensor1 = input_tensor.numpy().squeeze()

        if labels==0:        
            if heatmap0 is None:
                heatmap0 = output
                input_tensors0 = input_tensor1
            else:
                heatmap0 = np.vstack([heatmap0, output])
                input_tensors0 = np.vstack([input_tensors0, input_tensor1])
        if labels==1:        
            if heatmap1 is None:
                heatmap1 = output
                input_tensors1 = input_tensor1
            else:
                heatmap1 = np.vstack([heatmap1, output])
                input_tensors1 = np.vstack([input_tensors1, input_tensor1])
        if labels==2:        
            if heatmap2 is None:
                heatmap2 = output
                input_tensors2 = input_tensor1
            else:
                heatmap2 = np.vstack([heatmap2, output])
                input_tensors2 = np.vstack([input_tensors2, input_tensor1])
        if labels==3:        
            if heatmap3 is None:
                heatmap3 = output
                input_tensors3 = input_tensor1
            else:
                heatmap3 = np.vstack([heatmap3, output])
                input_tensors3 = np.vstack([input_tensors3, input_tensor1])
    if not os.path.exists(args.save_path):
        os.mkdir(f"{args.save_path}")
    os.mkdir(f"{args.save_path}{args.fold_name}")
    np.save(f"{args.save_path}{args.fold_name}/heatmap0.npy", heatmap0)
    np.save(f"{args.save_path}{args.fold_name}/heatmap1.npy", heatmap1)
    np.save(f"{args.save_path}{args.fold_name}/heatmap2.npy", heatmap2)
    np.save(f"{args.save_path}{args.fold_name}/heatmap3.npy", heatmap3)
    np.save(f"{args.save_path}{args.fold_name}/input_tensors0.npy", input_tensors0)
    np.save(f"{args.save_path}{args.fold_name}/input_tensors1.npy", input_tensors1)
    np.save(f"{args.save_path}{args.fold_name}/input_tensors2.npy", input_tensors2)
    np.save(f"{args.save_path}{args.fold_name}/input_tensors3.npy", input_tensors3)
    for i in range(len(heatmap0)):
        plot_and_save(heatmap0[i], input_tensors0[i], f"{args.save_path}{args.fold_name}/0_{i}.png")
    for i in range(len(heatmap1)):
        plot_and_save(heatmap1[i], input_tensors1[i], f"{args.save_path}{args.fold_name}/1_{i}.png")
    for i in range(len(heatmap2)):
        plot_and_save(heatmap2[i], input_tensors2[i], f"{args.save_path}{args.fold_name}/2_{i}.png")
    for i in range(len(heatmap3)):
        plot_and_save(heatmap3[i], input_tensors3[i], f"{args.save_path}{args.fold_name}/3_{i}.png")
    
    plot_and_save(np.mean(heatmap0, axis=0), np.mean(input_tensors0, axis=0), f"{args.save_path}{args.fold_name}/0_mean.png")
    plot_and_save(np.mean(heatmap1, axis=0), np.mean(input_tensors1, axis=0), f"{args.save_path}{args.fold_name}/1_mean.png")
    plot_and_save(np.mean(heatmap2, axis=0), np.mean(input_tensors2, axis=0), f"{args.save_path}{args.fold_name}/2_mean.png")
    plot_and_save(np.mean(heatmap3, axis=0), np.mean(input_tensors3, axis=0), f"{args.save_path}{args.fold_name}/3_mean.png")


# ============================================
    # input_tensors = np.load("data/data_war_sngp/C5/PET.npy")
    # heatmap = None
    # for i in range(input_tensors.shape[0]):
    #     input_tensor = input_tensors[i,:]
    #     input_tensor = torch.from_numpy(input_tensor).unsqueeze(dim=0).unsqueeze(dim=0)
    #     input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    #     print(input_tensor.size)
    #     output = net(input_tensor)
    #     input_tensor1 = input_tensor.numpy().squeeze()
    #     print(input_tensor.shape)
    #     if heatmap is None:
    #         heatmap = output
    #     else:
    #         heatmap = np.vstack([heatmap, output])



    # print(heatmap.shape)
    # # print(input_tensors.shape)
    # plot_and_save(np.mean(heatmap, axis=0), np.mean(input_tensors, axis=0), "logFile/EXP5/cam_PET.png")

# ============================================



