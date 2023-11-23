import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_block(nn.Module):
    def __init__(self):
        super(Inception_block, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],1)

class Net_Inception(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential()
        self.conv1.add_module("Conv1", nn.Conv1d(in_channels=1, out_channels=8, kernel_size=10))
        self.conv1.add_module("relu1", nn.ReLU())
        self.conv1.add_module("maxpool1", nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential()
        self.conv2.add_module("Conv2", nn.Conv1d(in_channels=8, out_channels=16, kernel_size=10))
        self.conv2.add_module("relu2", nn.ReLU())
        self.conv2.add_module("maxpool2", nn.MaxPool1d(kernel_size=2, stride=2))
        self.inception_block = Inception_block()



        self.fc1 = nn.Linear(31776,128)
        self.out = nn.Linear(128,2)
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(f"before inception: {x.shape}")
        x = self.inception_block(x)
        # print(f"after inception: {x.shape}")
        x = x.view(x.size(0),-1)
        # x = x.reshape(-1,256)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x,dim=1)
        return x

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.randn(32, 1, 4000).to(device)
    model = Net_Inception()
    model.to(device)
    model.eval()
    output = model(input_data)
    print(output.shape)
    # assert output.sahpe == (1,10), "Output shape is incorrect."