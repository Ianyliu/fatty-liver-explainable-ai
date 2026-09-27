import torch
from torch.nn import Sequential, Linear, BatchNorm1d, LayerNorm, ReLU
from torch_geometric.nn import global_mean_pool, global_max_pool, GINConv, GATConv, GATv2Conv
import torch.nn as nn
import torch.nn.functional as F
import torchvision, timm

class SETNET_GIN(torch.nn.Module):
    
    def __init__(self, num_layers, embed_dim, hidden_dim, output_dim):
        super(SETNET_GIN, self).__init__()

        self.num_layers = num_layers
        
        self.gnn = torch.nn.ModuleList()  
        self.gnn.append(
            GINConv(Sequential(Linear(embed_dim, hidden_dim), 
                               BatchNorm1d(hidden_dim), 
                               ReLU(),
                               Linear(hidden_dim, hidden_dim), ReLU())))
        
        for l in range(num_layers-1):
            
            self.gnn.append(
                GINConv(Sequential(Linear(hidden_dim, hidden_dim), 
                                   BatchNorm1d(hidden_dim), 
                                   ReLU(),
                                   Linear(hidden_dim, hidden_dim), 
                                   ReLU())))

        self.lin1 = Linear(hidden_dim, embed_dim)
        self.lin2 = Linear(embed_dim, output_dim)
        
    def forward(self, x, A, batch, train_mask):
        
        #x=self.cnn(x)
        
        for l in range(self.num_layers):
            x = x*train_mask
            x = self.gnn[l](x, A)
        
        x = x*train_mask
        x = self.lin1(x).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x

    
class SETNET_GAT(torch.nn.Module):
    
    def __init__(self, num_layers, embed_dim, hidden_dim, output_dim):
        super(SETNET_GAT, self).__init__()

        self.num_layers = num_layers
        
        self.mlp = torch.nn.ModuleList()
        self.gnn = torch.nn.ModuleList()
        
        #self.gnn.append(GATv2Conv(embed_dim, hidden_dim))
        self.gnn.append(GATConv(embed_dim, hidden_dim))
        
        self.mlp.append(Sequential(
            Linear(hidden_dim, hidden_dim), 
            BatchNorm1d(hidden_dim), 
            ReLU(),
            Linear(hidden_dim, hidden_dim), 
            ReLU()))
        
        for l in range(num_layers-1):
            
            #self.gnn.append(GATv2Conv(hidden_dim, hidden_dim))
            self.gnn.append(GATConv(hidden_dim, hidden_dim))
            
            self.mlp.append(Sequential(
                Linear(hidden_dim, hidden_dim), 
                BatchNorm1d(hidden_dim), 
                ReLU(),
                Linear(hidden_dim, hidden_dim), 
                ReLU()))

        self.lin1 = Linear(hidden_dim, embed_dim)
        self.lin2 = Linear(embed_dim, output_dim)
        
    def forward(self, x, A, batch, train_mask):
        
        #x=self.cnn(x)
        
        for l in range(self.num_layers):
            
            x = x*train_mask
            x = self.gnn[l](x, A)
            x = x*train_mask
            x = self.mlp[l](x)
            
        x = x*train_mask        
        x = self.lin1(x).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x


class SETNET_MLP(torch.nn.Module):
    
    def __init__(self, num_layers, embed_dim, hidden_dim, output_dim):
        super(SETNET_MLP, self).__init__()

        self.num_layers = num_layers
        
        self.mlp = torch.nn.ModuleList()
        
        self.mlp.append(Sequential(
            Linear(embed_dim, hidden_dim), 
            LayerNorm(hidden_dim), 
            ReLU(),
            Linear(hidden_dim, hidden_dim), 
            ReLU()))
        
        for l in range(num_layers-1):
            
            self.mlp.append(Sequential(
                Linear(hidden_dim, hidden_dim), 
                LayerNorm(hidden_dim), 
                ReLU(),
                Linear(hidden_dim, hidden_dim), 
                ReLU()))

        
        self.lin1 = Linear(hidden_dim, embed_dim)
        self.lin2 = Linear(embed_dim, output_dim)

        
    def forward(self, x, A, batch, train_mask):
        
        #x=self.cnn(x)
        
        for l in range(self.num_layers):

            x = self.mlp[l](x)
                
        x = self.lin1(x).relu()
        x = global_max_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x


class SETNET_MAX(torch.nn.Module):
    
    def __init__(self, num_layers, embed_dim, hidden_dim, output_dim):
        super(SETNET_MAX, self).__init__()

        self.num_layers = num_layers
        
        self.mlp = torch.nn.ModuleList()
        
        self.mlp.append(Sequential(
            Linear(embed_dim, hidden_dim), 
            LayerNorm(hidden_dim), 
            ReLU(),
            Linear(hidden_dim, hidden_dim), 
            ReLU()))
        
        for l in range(num_layers-1):
            
            self.mlp.append(Sequential(
                Linear(hidden_dim, hidden_dim), 
                LayerNorm(hidden_dim), 
                ReLU(),
                Linear(hidden_dim, hidden_dim), 
                ReLU()))

        
        self.lin1 = Linear(hidden_dim, embed_dim)
        self.lin2 = Linear(embed_dim, output_dim)

        
    def forward(self, x, A, batch, train_mask):
        
        #x=self.cnn(x)
        
        #for l in range(self.num_layers):

        #    x = self.mlp[l](x)
                
        #x = self.lin1(x).relu()
        x = global_max_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x


def encoder_model(name, input_dim, num_layers, num_classes, device):

    pretrained_encoder = None
    
    if name=="SETNET_GIN":
        
        encoder=SETNET_GIN(num_layers=num_layers, embed_dim=input_dim, hidden_dim=512, output_dim=num_classes)
        
    if name=="SETNET_MLP":
        
        encoder=SETNET_MLP(num_layers=num_layers, embed_dim=input_dim, hidden_dim=512, output_dim=num_classes)

    if name=="SETNET_MAX":
        
        encoder=SETNET_MAX(num_layers=num_layers, embed_dim=input_dim, hidden_dim=512, output_dim=num_classes)
    
    if name=="SETNET_GAT":
        
        encoder=SETNET_GAT(num_layers=num_layers, embed_dim=input_dim, hidden_dim=512, output_dim=num_classes)

    return encoder.to(device), pretrained_encoder






def image_encoder_model(name, pretrained, num_classes, device):
    
    encoder = None
    pretrained_encoder = None
        
        if pretrained==True:
            
            pretrained_encoder=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            pretrained_encoder.fc = nn.Identity()
        else: 
            pretrained_encoder=None
            encoder=torchvision.models.resnet50(pretrained=False, num_classes=num_classes)
        
   
    if name=="densenet121":
        
        if pretrained==True:
            
            pretrained_encoder=torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
            pretrained_encoder.classifier = nn.Identity()
            
            
        else:
            
            pretrained_encoder=None
            encoder=torchvision.models.densenet121(pretrained=False)
            encoder.classifier=nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    
    if name=="vitl16in21k":
        
        if pretrained==True:
            
            pretrained_encoder=timm.create_model('vit_large_patch16_224_in21k', pretrained=True)
            pretrained_encoder.head = nn.Identity()
            
        else:
            
            pretrained_encoder=None
            encoder=torchvision.models.vit_b_16(pretrained=False)
            encoder.heads=nn.Linear(in_features=768, out_features=num_classes, bias=True)
    
    if pretrained_encoder!=None:
        
        pretrained_encoder=pretrained_encoder.to(device)
        
        for param in pretrained_encoder.parameters():
            param.requires_grad = False 
    
    return encoder, pretrained_encoder.to(device)
