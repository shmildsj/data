import os.path as osp
import argparse
import os
import pandas as pd
from torch_geometric.data import Data
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from spline import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops)
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv,Sequential
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
import argparse
import os.path as osp
import time

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, APPNP, GCNConv

EPS = 1e-15
MAX_LOGSTD = 10
MAX_TEMP = 2.0 
MIN_TEMP = 0.1 

sc = 0.8 

decay_weight = np.log(MAX_TEMP/MIN_TEMP)
decay_step = 150.0
patience = 40

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DVGAE')
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=800)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# transform = T.Compose([
#     T.NormalizeFeatures(),
#     T.ToDevice(device),
#     T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
#                       split_labels=True, add_negative_train_samples=False),
# ])
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
#dataset = Planetoid(path, args.dataset, transform=transform)
#train_data, val_data, test_data = dataset[0]

interaction = pd.read_csv("D:/pycharm/d-vgae-main/data/RNADisease/association_matrix.csv", index_col=0)
train_dataset = pd.read_csv('D:/pycharm/d-vgae-main/data/RNADisease/train.txt', header=0, sep=',').values
test_dataset = pd.read_csv('D:/pycharm/d-vgae-main/data/RNADisease/test.txt', header=0, sep=',').values
d_feature = pd.read_csv("D:/pycharm/d-vgae-main/data/RNADisease/DD_matrix.csv").sort_values('snoRNA_id').iloc[:, 1:]  # 471
m_feature = pd.read_csv("D:/pycharm/d-vgae-main/data/RNADisease/MM_matrix.csv").sort_values('disease_id').iloc[:, 1:]  # 84

# interaction = pd.read_csv("D:/pycharm/d-vgae-main/data/indepent/association_matrix.csv", index_col=0)
# train_dataset = pd.read_csv('D:/pycharm/d-vgae-main/data/indepent/train.txt', header=0, sep=',').values
# test_dataset = pd.read_csv('D:/pycharm/d-vgae-main/data/indepent/test.txt', header=0, sep=',').values
# d_feature = pd.read_csv("D:/pycharm/d-vgae-main/data/indepent/DD_matrix.csv").sort_values('snoRNA_id').iloc[:, 1:]  # 471
# m_feature = pd.read_csv("D:/pycharm/d-vgae-main/data/indepent/MM_matrix.csv").sort_values('disease_id').iloc[:, 1:]  # 84

d_emb = torch.FloatTensor(d_feature.values)
m_emb = torch.FloatTensor(m_feature.values)
d_emb = torch.cat([d_emb, torch.zeros(d_emb.size(0), max(d_emb.size(1), m_emb.size(1)) - d_emb.size(1))], dim=1)
m_emb = torch.cat([m_emb, torch.zeros(m_emb.size(0), max(d_emb.size(1), m_emb.size(1)) - m_emb.size(1))], dim=1)

feature = torch.cat([d_emb, m_emb])

train_edge_index = []
for i in train_dataset:
    train_edge_index.append([i[0], i[1] + len(d_feature)])
train = Data(x=feature, edge_index=torch.LongTensor(train_edge_index).T)

train_data, _, _ = T.RandomLinkSplit(num_val=0., num_test=0.,
                                     is_undirected=True,
                                     split_labels=True,
                                     add_negative_train_samples=True)(train)

test_edge_index = []
for i in test_dataset:
    test_edge_index.append([i[0], i[1] + len(d_feature)])
test = Data(x=feature, edge_index=torch.LongTensor(test_edge_index).T)

test_data, _, _ = T.RandomLinkSplit(num_val=0., num_test=0.,
                                    is_undirected=True,
                                    split_labels=True,
                                    add_negative_train_samples=True)(test)
m, d = interaction.values.nonzero()
adj = torch.LongTensor([d, m + len(d_feature)])
dataset = Data(x=feature, edge_index=adj)

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0.6, add_self_loops=True):
        super(GATLayer, self).__init__()
        # 初始化 GATConv 层
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,             # 注意力机制的头数
            concat=concat,           # 是否连接多头的输出
            negative_slope=negative_slope,  # LeakyReLU中的负斜率参数
            dropout=dropout,         # 注意力系数的dropout
            add_self_loops=add_self_loops  # 是否添加自环
        )

    def forward(self, x, edge_index):
        # 应用 GAT 层
        x = self.gat(x, edge_index)
        return x

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, cached=False, add_self_loops=True, normalize=True, bias=True):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels, improved=improved, cached=cached, add_self_loops=add_self_loops, normalize=normalize, bias=bias)

    def forward(self, x, edge_index):
        # Apply the GCN layer
        x = self.gcn(x, edge_index)
        return x

class GINLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINLayer, self).__init__()
        # 定义 MLP 作为 GINConv 的更新函数
        nn1 = Sequential('x', [
            (nn.Linear(in_channels, out_channels), 'x -> x'),
            (nn.ReLU(), 'x -> x'),
            (nn.Linear(out_channels, out_channels), 'x -> x')
        ])
        # 初始化 GINConv 层
        self.gin = GINConv(nn1, train_eps=True)

    def forward(self, x, edge_index):
        # 应用 GIN 层
        x = self.gin(x, edge_index)
        return x

class SAGELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGELayer, self).__init__()
        # 初始化 SAGEConv 层
        self.sage = SAGEConv(
            in_channels=in_channels,
            out_channels=out_channels,
            normalize=True,  # 是否应用L2归一化
            root_weight=True,  # 是否包含自连接权重
            bias=True  # 是否加入偏置项
        )

    def forward(self, x, edge_index):
        # 应用 GraphSAGE 层
        x = self.sage(x, edge_index)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index,not_prop=0):

        if args.model == 'DVGAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)

            x = self.linear2(x)
            x = F.normalize(x,p=2,dim=1) * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x, x_
# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, edge_index):
#         super(Encoder, self).__init__()
#         self.linear1 = nn.Linear(in_channels, out_channels)
#         self.kan_layer = KANLayer(in_dim=out_channels, out_dim=out_channels)  # 调整KANLayer参数以符合您的需求
#         self.linear2 = nn.Linear(out_channels, out_channels)  # 确保输出维度一致
#         self.propagate = APPNP(K=1, alpha=0)
#
#     def forward(self, x, edge_index, not_prop=0):
#         if args.model == 'DVGAE':
#             x = self.linear1(x)
#             x = self.kan_layer(x)  # 在第一层和第二层线性变换之间应用KANLayer
#             x_ = self.propagate(x, edge_index)  # 应用APPNP传播
#
#             x = self.linear2(x)
#             x = F.normalize(x, p=2, dim=1) * args.scaling_factor
#             x = self.propagate(x, edge_index)
#             return x, x_

# class Encoder2(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, edge_index):
#         super(Encoder2, self).__init__()
#         self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
#         self.linear2 = nn.Linear(in_channels, out_channels, bias=False)
#
#     def forward(self, x, edge_index,not_prop=0):
#
#         if args.model == 'DVGAE':
#             x_ = self.linear1(x)
#             x = self.linear2(x)
#             x = F.normalize(x,p=2,dim=1) * sc
#             return x, x_

# class Encoder2(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, edge_index):
#         super(Encoder2, self).__init__()
#         self.linear1 = nn.Linear(in_channels, out_channels, bias=False)  # 确保输出维度一致
#         self.kan_layer = KANLayer(in_dim=in_channels, out_dim=out_channels)  # 调整KANLayer的参数以符合您的需求
#         #self.linear2 = nn.Linear(in_channels, out_channels, bias=False)  # 确保输出维度一致
#
#     def forward(self, x, edge_index, not_prop=0):
#         if args.model == 'DVGAE':
#             x_ = self.linear1(x)
#             x = self.kan_layer(x)  # 应用KANLayer处理第一个线性层的输出
#             #x = self.linear2(x)
#             x = F.normalize(x, p=2, dim=1) * sc  # 应用缩放因子
#             return x,x_

class Encoder2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(Encoder2, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)  # 确保输出维度一致
        self.kan = KANLayer(in_channels, out_channels)  # 调整KANLayer的参数以符合您的需求
        #self.linear2 = nn.Linear(in_channels, out_channels, bias=False)  # 确保输出维度一致

    def forward(self, x, edge_index, not_prop=0):
        if args.model == 'DVGAE':
            x_ = self.linear1(x)
            x = self.kan(x)  # 应用KANLayer处理第一个线性层的输出
            #x = self.linear2(x)
            x = F.normalize(x, p=2, dim=1) * sc  # 应用缩放因子
            return x,x_

class DVGAE(torch.nn.Module):
    def __init__(self, encoder1, encoder2, decoder):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder = InnerProductDecoder2() if decoder is None else decoder
        DVGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder1)
        reset(self.encoder2)
        reset(self.decoder)
  
    def encode1(self, *args, **kwargs):
        """"""
        self.__mu1__, self.__logstd1__ = self.encoder1(*args, **kwargs)
        self.__logstd1__ = self.__logstd1__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu1__, self.__logstd1__)
        return z
    def encode2(self, *args, **kwargs):
        """"""
        self.__mu2__, self.__logstd2__ = self.encoder2(*args, **kwargs)
        self.__logstd2__ = self.__logstd2__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu2__, self.__logstd2__)
        return z
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu 
    
    # def test(self, z1, z2, temp, pos_edge_index, neg_edge_index):
    #
    #     pos_y = z1.new_ones(pos_edge_index.size(1))
    #     neg_y = z1.new_zeros(neg_edge_index.size(1))
    #     y = torch.cat([pos_y, neg_y], dim=0)
    #
    #     pos_pred = self.decoder(z1, z2, temp, pos_edge_index, sigmoid=True, training=False)
    #     neg_pred = self.decoder(z1, z2, temp, neg_edge_index, sigmoid=True, training=False)
    #     pred = torch.cat([pos_pred, neg_pred], dim=0)
    #
    #     y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
    #
    #     return roc_auc_score(y, pred), average_precision_score(y, pred)

    def test(self, z1, z2, temp, pos_edge_index, neg_edge_index):

        # Generate positive and negative labels
        pos_y = z1.new_ones(pos_edge_index.size(1))
        neg_y = z1.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        # Generate predictions
        pos_pred = self.decoder(z1, z2, temp, pos_edge_index, sigmoid=True, training=False)
        neg_pred = self.decoder(z1, z2, temp, neg_edge_index, sigmoid=True, training=False)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        # Detach and move data to CPU for sklearn processing
        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        pred_binary = (pred > 0.5).astype(int)  # Converting probabilities to binary output for some metrics

        # Calculate metrics
        auc = roc_auc_score(y, pred)
        aupr = average_precision_score(y, pred)
        acc = accuracy_score(y, pred_binary)
        pre = precision_score(y, pred_binary)
        sen = recall_score(y, pred_binary)  # Sensitivity is the same as recall
        f1 = f1_score(y, pred_binary)

        # Return all metrics
        return auc, aupr, acc, pre, sen, f1
  
    def recon_loss(self, z1, z2, temp, pos_edge_index, neg_edge_index=None):

        decode_p = self.decoder(z1, z2, temp, pos_edge_index, sigmoid=True, training=True)
        pos_loss = -torch.log(decode_p + EPS).sum()

        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z1.size(0))
        
        decode_n1 = self.decoder(z1, z2, temp, neg_edge_index, sigmoid=True, training=True)
        neg_loss = -torch.log(1 -decode_n1 + EPS).sum() 

        return (pos_loss + neg_loss) 
    
    def kl_loss1(self, mu=None, logstd=None):

        mu = self.__mu1__ if mu is None else mu
        logstd = self.__logstd1__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        
    def kl_loss2(self, mu=None, logstd=None):

        mu = self.__mu2__ if mu is None else mu
        logstd = self.__logstd2__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    
class InnerProductDecoder2(torch.nn.Module):
    def __init__(self):
        super().__init__()        

    def forward(self, z1, z2,  temp, edge_index, sigmoid=True, training=True):
            
        if training: 
            z11 = z1.detach().clone()
            vf = (z11[edge_index[0]] * z11[edge_index[1]]).sum(dim=1) 
            la = torch.cat(  (torch.unsqueeze(vf, 1), torch.zeros(torch.unsqueeze(vf, 1).shape).to(device)   ),1)
            la_ra = la
            a = F.gumbel_softmax((la_ra), tau=temp, hard=True)[:,:1]
            value_feature = (z1[edge_index[0]] * z1[edge_index[1]]).sum(dim=1)
            value_network =  z2[edge_index[0],[0]] + z2[edge_index[1],[0]]
            feature_flag = torch.flatten(a)
            return feature_flag*torch.sigmoid(value_feature) + (1-feature_flag)*torch.sigmoid(value_network) if sigmoid else value
            
        else:
            z11 = z1.detach().clone()
            vf = (z11[edge_index[0]] * z11[edge_index[1]]).sum(dim=1)
            la = torch.cat(  (torch.unsqueeze(vf, 1), torch.zeros(torch.unsqueeze(vf, 1).shape).to(device)   ),1)
            la_ra = la
            a = F.softmax((la_ra), dim=1)[:,:1]
            value_feature = (z1[edge_index[0]] * z1[edge_index[1]]).sum(dim=1)
            value_network =  z2[edge_index[0],[0]] + z2[edge_index[1],[0]]         
            return torch.sigmoid(value_feature)*torch.sigmoid(value_feature) + (1-torch.sigmoid(value_feature))*torch.sigmoid(value_network) if sigmoid else value

    def forward_all(self, z, sigmoid=True):

        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

class KANLayer(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0,
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True,
                 save_plot_data=True, device='cpu', sparse_init=False):
        super(KANLayer, self).__init__()
        # size
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(self.in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 1 / 2) * noise_scale / num
        # shape: (size, coef)
        self.coef = torch.nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k))
        # if isinstance(scale_base, float):
        if sparse_init:
            mask = sparse_mask(in_dim, out_dim)
        else:
            mask = 1.

        self.scale_base = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_base * mask).requires_grad_(
            sb_trainable)  # make scale trainable

        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * mask).requires_grad_(
            sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        self.grid_eps = grid_eps

    def forward(self, x):
        batch = x.shape[0]

        preacts = x[:, None, :].clone().expand(batch, self.out_dim, self.in_dim)

        base = self.base_fun(x)  # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)  # y shape: (batch, in_dim, out_dim)

        postspline = y.clone().permute(0, 2, 1)  # postspline shape: (batch, out_dim, in_dim)

        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * y
        y = self.mask[None, :, :] * y

        postacts = y.clone().permute(0, 2, 1)

        y = torch.sum(y, dim=1)  # shape (batch, out_dim)
        # return y, preacts, postacts, postspline
        return y

    def update_grid_from_samples(self, x, mode='sample'):
        batch = x.shape[0]
        # x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
            grid_uniform = grid_adaptive[:, [0]] + h * torch.arange(num_interval + 1, )[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        grid = get_grid(num_interval)

        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)

        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        batch = x.shape[0]

        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
            grid_uniform = grid_adaptive[:, [0]] + h * torch.arange(num_interval + 1, )[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        grid = get_grid(num_interval)

        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)

        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def get_subset(self, in_id, out_id):
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:, out_id]
        spb.scale_base.data = self.scale_base[in_id][:, out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:, out_id]
        spb.mask.data = self.mask[in_id][:, out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb

    def swap(self, i1, i2, mode='in'):

        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:, i1], data[:, i2] = data[:, i2].clone(), data[:, i1].clone()

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

channels = args.channels    
in_channels= dataset.num_features
N = train_data.x.shape[0]

model = DVGAE(Encoder(in_channels, channels, train_data.edge_index), Encoder2(N, 2, train_data.edge_index) , InnerProductDecoder2()).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

network_input = torch.eye(N).to(device)
l1 = train_data.edge_index

def train(epoch):
    temp = np.maximum(MAX_TEMP*np.exp(-(epoch-1)/decay_step*decay_weight), MIN_TEMP)

    model.train()
    optimizer.zero_grad()

    x = train_data.x.to(device)
    l1 = train_data.edge_index.to(device)
    z1 = model.encode1( x , l1)
    z2 = model.encode2(network_input, l1)
    loss = model.recon_loss(z1,z2, temp, train_data.pos_edge_label_index)

    loss = loss + (1.0 / N) * (model.kl_loss1()+model.kl_loss2())
    loss.backward()
    optimizer.step()
    return loss

def test(pos_edge_index, neg_edge_index, selected_list, plot_his=0):
    model.eval()
    with torch.no_grad():
        x = train_data.x.to(device)
        selected_list = selected_list.to(device)
        z1 = model.encode1(x, selected_list)
        z2 = model.encode2(network_input, selected_list)
    return model.test(z1, z2, 1.0, pos_edge_index, neg_edge_index)

early_stopping = EarlyStopping(patience = patience, verbose = True)
for run in range(5):
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        loss = float(loss)

        with torch.no_grad():
            val_pos, val_neg = test_data.pos_edge_label_index, test_data.neg_edge_label_index
            auc, aupr, acc, pre, sen, f1 = test(val_pos, val_neg, train_data.edge_index)
            if epoch > 200:  # after minimum epoch, check early stopping
                early_stopping(-auc, model)
                if early_stopping.early_stop:
                    break

    model.load_state_dict(torch.load('checkpoint.pt'))

    test_pos, test_neg = test_data.pos_edge_label_index, test_data.neg_edge_label_index
    auc, aupr, acc, pre, sen, f1 = test(test_pos, test_neg, train_data.edge_index)

    print(
        'Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f} AUPR: {:.4f} ACC: {:.4f} PRE: {:.4f} SEN: {:.4f} F1: {:.4f} '.format(
            epoch, loss, auc, aupr, acc, pre, sen, f1))

