import torch.nn as nn
from dgl.nn import GATConv
import torch.nn.functional as F
import torch as th

import math

class CaseLink(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout, layer_num, num_heads):
        super(CaseLink, self).__init__()
        self.hidden_size = h_dim
        self.layer_num = layer_num
        self.layers = nn.ModuleList()
        self.GATconv1 = GATConv(in_feats=in_dim, out_feats=h_dim, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True)
        self.GATconv2 = GATConv(in_feats=h_dim, out_feats=out_dim, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True)
        self.GATconv3 = GATConv(in_feats=h_dim, out_feats=out_dim, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True)
        self.reset_parameters()

    def forward(self, g, in_feat):
        if self.layer_num == 3:
            h = self.GATconv1(g, in_feat)
            h = F.relu(h)
            h = th.mean(h, dim=1)

            h = self.GATconv2(g, h)    
            h = F.relu(h) 
            h = th.mean(h, dim=1)   

            h = self.GATconv3(g, h)        
        
        elif self.layer_num == 2:
            h = self.GATconv1(g, in_feat)
            h = F.relu(h)
            h = th.mean(h, dim=1)
            
            h = self.GATconv2(g, h)
        
        elif self.layer_num == 1:
            h = self.GATconv1(g, in_feat)            
            
        h = th.mean(h, dim=1)+in_feat

        return h 

    def reset_parameters(self):
        if self.hidden_size == 0:
            stdv = 1.0 / math.sqrt(self.in_dim)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)
        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

def early_stopping(highest_f1score, epoch_f1score, epoch_num, continues_epoch):
    if epoch_f1score <= highest_f1score:
        if continues_epoch > 20:
            return [highest_f1score, True]
        else:
            continues_epoch += 1
            return [highest_f1score, False, continues_epoch]
    else:
        continues_epoch = 0
        return [epoch_f1score, False, continues_epoch]
    

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = th.sqrt(self.mse(yhat,y) + self.eps)
        return loss

