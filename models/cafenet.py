import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class CaFeNet(nn.Module):
    def __init__(self,num_classes, lr = 0.001):
        super().__init__()
        self.feature_extractor = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor.classifier = nn.Identity() #identity: input shape = output shape
        self.feature_dim = 1280
        self.num_classes = num_classes
        self.attention_dim = self.feature_dim
        self.attn = CentroidAttention(num_classes,self.feature_dim,self.attention_dim)
        self.classifier = nn.Linear(self.feature_dim*2,num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x, labels = None):
        features = self.feature_extractor(x)
        features = self.attn(features,labels)
        logits = self.classifier(features)
        return logits, features
    
    def print_num_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params_millions = total_params / 1e6
        print(f"Total trainable params: {total_params_millions:.3g} M")
        
        
class CentroidAttention(nn.Module):
    def __init__(self, num_classes, feature_dim, attention_dim, qkv_bias = False, attn_drop = 0, proj_drop = 0):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.scale = attention_dim ** -0.5
        
        self.register_buffer('centers', torch.zeros(num_classes,feature_dim))
        self.register_buffer('center_values', torch.zeros(num_classes,feature_dim))
        self.register_buffer('center_counts', torch.zeros(num_classes,1))
        
        self.query = nn.Linear(feature_dim,attention_dim, bias=qkv_bias)
        self.key = nn.Linear(feature_dim, attention_dim, bias=qkv_bias)
        self.value = nn.Linear(feature_dim,attention_dim,bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        if self.feature_dim != self.attention_dim:
            self.proj = nn.Linear(self.attention_dim, self.feature_dim)
            self.proj_drop = nn.Dropout(proj_drop)
        else:
            self.proj = nn.Identity()
            self.proj_drop = nn.Identity()
            
    def update_center(self,features, labels):
        if labels is not None:
            with torch.no_grad():
                for i in torch.unique(labels):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        class_features = features[idx]
                        class_sum = class_features.sum(dim=0)
                        self.center_values[i] += class_sum
                        self.center_counts[i] += idx.sum()
                        
                        self.centers[i] = self.center_values[i] / self.center_counts[i]
        else:
            self.center_values.zero_()
            self.center_counts.zero_()
            
        return self.centers
    
    def forward(self,features, labels=None):
        centers = self.update_center(features=features, labels=labels)
        
        q = self.query(features) #features.shape = (batch_size, attention_dim)
        k = self.key(centers)    #centers.shape = (num_classes, attention_dim)
        
        scores = torch.matmul(q, k.t()) * self.scale
        scores = torch.softmax(scores, dim=-1)
        scores = self.attn_drop(scores)
        
        v = self.value(centers)
        attention_values = torch.matmul(scores,v)
        attention_values = self.proj(attention_values)
        attention_values = self.proj_drop(attention_values)
        
        output = torch.cat([features, attention_values], dim=-1)
        
        return output