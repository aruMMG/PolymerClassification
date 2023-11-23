import torch.nn as nn
import torch
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.n_class = num_class
        self.fc1 = nn.Linear(4000,2000)
        self.bn1 = nn.BatchNorm1d(num_features=2000, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2000,100)
        self.bn2 = nn.BatchNorm1d(num_features=100, momentum=0.1)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(100,self.n_class)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.out(x)
        return x

class Net(nn.Module):
    def __init__(self, num_class=2, in_channels=1):
        super().__init__()
        self.n_class = num_class
        self.in_channels = in_channels
        self.conv1 = nn.Sequential()
        self.conv1.add_module("Conv1", nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3))
        self.conv1.add_module("relu1", nn.ReLU())
        self.conv1.add_module("maxpool1", nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential()
        self.conv2.add_module("Conv2", nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3))
        self.conv2.add_module("relu2", nn.ReLU())
        self.conv2.add_module("maxpool2", nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential()
        self.conv3.add_module("Conv3", nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3))
        self.conv3.add_module("relu3", nn.ReLU())
        self.conv3.add_module("maxpool3", nn.MaxPool1d(kernel_size=5, stride=5))
        self.conv4 = nn.Sequential()
        self.conv4.add_module("Conv4", nn.Conv1d(in_channels=148, out_channels=256, kernel_size=7, padding=3))
        self.conv4.add_module("relu4", nn.ReLU())
        self.conv4.add_module("maxpool4", nn.MaxPool1d(kernel_size=5, stride=5))

        self.fc1 = nn.Linear(256*40,128)
        self.out = nn.Linear(128,self.n_class)
    def forward(self, x):
        x1 = x
        x = self.conv1(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x1 = x1.reshape(x.shape[0],-1,200)
        # print(x1.shape)
        x = torch.cat((x,x1),dim=1)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(-1, 256 * 40)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x,dim=1)
        return x

# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    # 2. Initialize the class with appropriate variables
    def __init__(self, 
                 in_channels:int=1,
                 patch_size:int=20,
                 embedding_dim:int=20):
        super().__init__()
        
        self.patch_size = patch_size
        self.patcher = nn.Conv1d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        # 3. Create a layer to turn an image into patches
        # self.patcher = nn.Conv2d(in_channels=in_channels,
        #                          out_channels=embedding_dim,
        #                          kernel_size=patch_size,
        #                          stride=patch_size,
        #                          padding=0)

    # 5. Define the forward method 
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        
        # Perform the forward pass
        x_patched = self.patcher(x)
        # print(x_patched.shape)
        # 6. Make sure the output shape has the right order 
        return x_patched.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, 

class SpectroscopyTransformerEncoder(nn.Module): 
  def __init__(self,
               input_size=4000, # from Table 3
               num_channels=1,
               patch_size=20,
               embedding_dim=20, # from Table 1
               dropout=0.1, 
               mlp_size=256, # from Table 1
               num_transformer_layers=3, # from Table 1
               num_heads=4, # from Table 1 (number of multi-head self attention heads)
               num_classes=2): # generic number of classes (this can be adjusted)
    super().__init__()

    # Assert image size is divisible by patch size 
    assert input_size % patch_size == 0, "Image size must be divisble by patch size."

    # 1. Create patch embedding
    self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)

    # 2. Create class token
    self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                    requires_grad=True)

    # 3. Create positional embedding
    num_patches = input_size // patch_size
    self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))

    # 4. Create patch + position embedding dropout 
    self.embedding_dropout = nn.Dropout(p=dropout)

    # # 5. Create Transformer Encoder layer (single)
    # self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
    #                                                             nhead=num_heads,
    #                                                             dim_feedforward=mlp_size,
    #                                                             activation="gelu",
    #                                                             batch_first=True,
    #                                                             norm_first=True)

    # 5. Create stack Transformer Encoder layers (stacked single layers)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                              nhead=num_heads,
                                                                                              dim_feedforward=mlp_size,
                                                                                              activation="gelu",
                                                                                              batch_first=True,
                                                                                              norm_first=True), # Create a single Transformer Encoder Layer
                                                     num_layers=num_transformer_layers) # Stack it N times

    # 7. Create MLP head
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim,
                  out_features=num_classes)
    )

  def forward(self, x):
    # Get some dimensions from x
    batch_size = x.shape[0]

    # Create the patch embedding
    x = self.patch_embedding(x)
    # print(x.shape)

    # First, expand the class token across the batch size
    class_token = self.class_token.expand(batch_size, -1, -1) # "-1" means infer the dimension

    # Prepend the class token to the patch embedding
    x = torch.cat((class_token, x), dim=1)
    # print(x.shape)

    # Add the positional embedding to patch embedding with class token
    x = self.positional_embedding + x
    # print(x.shape)

    # Dropout on patch + positional embedding
    x = self.embedding_dropout(x)

    # Pass embedding through Transformer Encoder stack
    x = self.transformer_encoder(x)

    # Pass 0th index of x through MLP head
    x = self.mlp_head(x[:, 0])

    return x



class PreBlock(torch.nn.Module):
    """
    Preprocessing module. It is designed to replace filtering and baseline correction.

    Args:
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
    """
    def __init__(self):
        super().__init__()
        self.pool1 = torch.nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.pool2 = torch.nn.AvgPool1d(kernel_size=13, stride=1, padding=6)
        self.pool3 = torch.nn.AvgPool1d(kernel_size=7, stride=1, padding=3)
        self.ln_1 = torch.nn.LayerNorm(4000)

    def forward(self, x):

        x = x.squeeze(dim=1)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.ln_1(x)
        x = x.unsqueeze(dim=1)

        return x

class SpectroscopyTransformerEncoder_PreT(nn.Module):
    """
    fNIRS-PreT model

    Args:
        n_class: number of classes.
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
        dim: last dimension of output tensor after linear transformation.
        depth: number of Transformer blocks.
        heads: number of the multi-head self-attention.
        mlp_dim: dimension of the MLP layer.
        pool: MLP layer classification mode, 'cls' is [CLS] token pooling, 'mean' is  average pooling, default='cls'.
        dim_head: dimension of the multi-head self-attention, default=64.
        dropout: dropout rate, default=0.
        emb_dropout: dropout for patch embeddings, default=0.
    """
    def __init__(self,
               input_size=4000, # from Table 3
               num_channels=1,
               patch_size=20,
               embedding_dim=20, # from Table 1
               dropout=0.1, 
               mlp_size=256, # from Table 1
               num_transformer_layers=3, # from Table 1
               num_heads=4, # from Table 1 (number of multi-head self attention heads)
               num_classes=2): # generic number of classes (this can be adjusted)
        super().__init__()
        self.pre = PreBlock()
        self.IR_PreT = SpectroscopyTransformerEncoder(input_size, num_channels, patch_size, embedding_dim, dropout, mlp_size, num_transformer_layers, num_heads, num_classes)


    def forward(self, x):
        x = self.pre(x)
        x = self.IR_PreT(x)
        return x

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.randn(32, 1, 4000).to(device)
    model = SpectroscopyTransformerEncoder_PreT()
    model.to(device)
    model.eval()
    output = model(input_data)
    print(output.shape)
    # assert output.sahpe == (1,10), "Output shape is incorrect."