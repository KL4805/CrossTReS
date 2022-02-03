import numpy as np
import argparse
from model import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import time
from collections import OrderedDict
from utils import *
import copy
from dgl.nn import GATConv
import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

# This file implements the full version of using region embeddings to select good source data. 
parser = argparse.ArgumentParser()
parser.add_argument('--scity', type=str, default='NY')
parser.add_argument('--tcity', type=str, default='DC')
parser.add_argument('--dataname', type=str, default='Taxi', help='Within [Bike, Taxi]')
parser.add_argument('--datatype', type=str, default='pickup', help='Within [pickup, dropoff]')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument("--model", type=str, default='STNet_nobn', help='Within [STResNet, STNet, STNet_nobn]')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=-1, help='Random seed. -1 means do not manually set. ')
parser.add_argument('--data_amount', type=int, default=0, help='0: full data, 30/7/3 correspond to days of data')
parser.add_argument('--sinneriter', type=int, default=5, help='Number of inner iterations (source) for meta learning')
parser.add_argument('--tinneriter', type=int, default=5, help='Number of inner iterations (target) for meta learning')
parser.add_argument('--innerlr', type=float, default=1e-3, help='Learning rate for inner loop of meta-learning')
parser.add_argument('--outeriter', type=int, default=20, help='Number of outer iterations for meta-learning')
parser.add_argument('--outerlr', type=float, default=1e-3, help='Learning rate for the outer loop of meta-learning')
parser.add_argument('--topk', type=int, default=15)
parser.add_argument('--mmd_w', type=float, default=0.5, help='mmd weight')
parser.add_argument('--et_w', type=float, default=1, help='edge type discriminator weight')
parser.add_argument("--ma_coef", type=float, default=0.8, help='Moving average parameter for source domain weights')
parser.add_argument("--weight_reg", type=float, default=1e-3, help="Regularizer for the source domain weights.")
parser.add_argument("--pretrain_iter", type=int, default=30, help='Pre-training iterations per pre-training epoch. ')
parser.add_argument("--log_name", type=str, default=None, help='Filename for log file')
parser.add_argument('--save_name', type=str, default=None, help='Filename for saving the model.')
parser.add_argument("--rt_weight", type=float, default = 0.001, help='weight for the regiontrans objective')
parser.add_argument('--rt_dict', type=str, default = 'poi', help='path of the dictionary for regiontrans')
args = parser.parse_args()

if args.seed != -1:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
gpu_available = torch.cuda.is_available()
if gpu_available:
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')

dataname = args.dataname 
scity = args.scity 
tcity = args.tcity 
datatype = args.datatype 
num_epochs = args.num_epochs 
start_time = time.time() 
print("Running CrossGTP Full, from %s to %s, %s %s experiments, with %d days of data, on %s model" % \
    (scity, tcity, dataname, datatype, args.data_amount, args.model)) 

# Load spatio temporal data
target_data = np.load("../data/%s/%s%s_%s.npy" % (tcity, dataname, tcity, datatype)) 
lng_target, lat_target = target_data.shape[1], target_data.shape[2] 
mask_target = target_data.sum(0) > 0
th_mask_target = torch.Tensor(mask_target.reshape(1, lng_target, lat_target)).to(device) 
print("%d valid regions in target" % np.sum(mask_target)) 
target_emb_label = masked_percentile_label(target_data.sum(0).reshape(-1), mask_target.reshape(-1))

source_data = np.load("../data/%s/%s%s_%s.npy" % (scity, dataname, scity, datatype)) 
lng_source, lat_source = source_data.shape[1], source_data.shape[2] 
mask_source = source_data.sum(0) > 0
th_mask_source = torch.Tensor(mask_source.reshape(1, lng_source, lat_source)).to(device) 
print("%d valid regions in source" % np.sum(mask_source)) 
source_emb_label = masked_percentile_label(source_data.sum(0).reshape(-1), mask_source.reshape(-1))


lag = [-6, -5, -4, -3, -2, -1]
source_data, smax, smin = min_max_normalize(source_data)
target_data, max_val, min_val = min_max_normalize(target_data)

# compute bias for the date
bias = 0
if args.scity == 'CHI':
    if args.tcity == 'DC':
        bias = 0
    elif args.tcity == 'BOS':
        bias = -6 * 24
elif args.scity == 'NY':
    if args.tcity == 'DC':
        if dataname == 'Bike':
            bias = 0
        elif dataname == 'Taxi':
            bias = -5 * 24
    elif args.tcity == 'BOS': 
        bias = -6 * 24

source_train_x, source_train_y, source_val_x, source_val_y, source_test_x, source_test_y = split_x_y(source_data, lag)
# we concatenate all source data
source_x = np.concatenate([source_train_x, source_val_x, source_test_x], axis = 0)
source_y = np.concatenate([source_train_y, source_val_y, source_test_y], axis = 0)
target_train_x, target_train_y, target_val_x, target_val_y, target_test_x, target_test_y = split_x_y(target_data, lag)
if args.data_amount != 0:
    target_train_x = target_train_x[-args.data_amount * 24:, :, :, :]
    target_train_y = target_train_y[-args.data_amount * 24:, :, :, :]
    if bias != 0:
        source_train_x = source_train_x[-args.data_amount * 24 + bias:bias, :, :, :]
        source_train_y = source_train_y[-args.data_amount * 24 + bias:bias, :, :, :]
    else:
        source_train_x = source_train_x[-args.data_amount * 24:, :, :, :]
        source_train_y = source_train_y[-args.data_amount * 24:, :, :, :]
print("Source split to: x %s, y %s" % (str(source_x.shape), str(source_y.shape)))
# print("val_x %s, val_y %s" % (str(source_val_x.shape), str(source_val_y.shape)))
# print("test_x %s, test_y %s" % (str(source_test_x.shape), str(source_test_y.shape))) 

print("Target split to: train_x %s, train_y %s" % (str(target_train_x.shape), str(target_train_y.shape)))
print("val_x %s, val_y %s" % (str(target_val_x.shape), str(target_val_y.shape)))
print("test_x %s, test_y %s" % (str(target_test_x.shape), str(target_test_y.shape)))



target_train_dataset = TensorDataset(torch.Tensor(source_train_x), torch.Tensor(source_train_y), torch.Tensor(target_train_x), torch.Tensor(target_train_y))
target_val_dataset = TensorDataset(torch.Tensor(target_val_x), torch.Tensor(target_val_y))
target_test_dataset = TensorDataset(torch.Tensor(target_test_x), torch.Tensor(target_test_y))
target_train_loader = DataLoader(target_train_dataset, batch_size = args.batch_size, shuffle = True)
target_val_loader = DataLoader(target_val_dataset, batch_size = args.batch_size)
target_test_loader = DataLoader(target_test_dataset, batch_size = args.batch_size)
source_test_dataset = TensorDataset(torch.Tensor(source_test_x), torch.Tensor(source_test_y))
source_test_loader = DataLoader(source_test_dataset, batch_size = args.batch_size)
source_dataset = TensorDataset(torch.Tensor(source_x), torch.Tensor(source_y))
source_loader = DataLoader(source_dataset, batch_size = args.batch_size, shuffle=True)

# load regiontrans dictionary
dict_path = "../src_baselines/rt_dict/%s_%s_%s" % (args.scity, args.tcity, args.rt_dict)
with open(dict_path, 'r') as infile:
    rt_dict = eval(infile.read()) 
# and transform to tensors
matching_indices = []
matching_weight = []
for i in range(lng_target * lat_target):
    lng_idx, lat_idx = idx_1d22d(i, (lng_target, lat_target))
    (match_lng_idx,  match_lat_idx), match_weight = rt_dict[(lng_idx, lat_idx)]
    matching_indices.append(idx_2d2id((match_lng_idx, match_lat_idx), (lng_source, lat_source)))
    matching_weight.append(match_weight)
matching_indices = torch.Tensor(matching_indices).long()
matching_weight = torch.Tensor(matching_weight).view(1, -1).to(device)

# Load auxiliary data: poi data
source_poi = np.load("../data/%s/%s_poi.npy" % (scity, scity))
target_poi = np.load("../data/%s/%s_poi.npy" % (tcity, tcity))
source_poi = source_poi.reshape(lng_source * lat_source, -1) # regions * classes 
target_poi = target_poi.reshape(lng_target * lat_target, -1) # regions * classes 
transform = TfidfTransformer()
source_norm_poi = np.array(transform.fit_transform(source_poi).todense())
transform = TfidfTransformer()
target_norm_poi = np.array(transform.fit_transform(target_poi).todense()) 

# Build graphs
source_prox_adj = add_self_loop(build_prox_graph(lng_source, lat_source)) 
target_prox_adj = add_self_loop(build_prox_graph(lng_target, lat_target)) 
source_road_adj = add_self_loop(build_road_graph(scity, lng_source, lat_source)) 
target_road_adj = add_self_loop(build_road_graph(tcity, lng_target, lat_target)) 
source_poi_adj, source_poi_cos = build_poi_graph(source_norm_poi, args.topk)
target_poi_adj, target_poi_cos = build_poi_graph(target_norm_poi, args.topk) 
source_poi_adj = add_self_loop(source_poi_adj) 
target_poi_adj = add_self_loop(target_poi_adj) 
source_s_adj, source_d_adj, source_od_adj = build_source_dest_graph(scity, dataname, lng_source, lat_source, args.topk)
target_s_adj, target_d_adj, target_od_adj = build_source_dest_graph(tcity, dataname, lng_target, lat_target, args.topk) 
source_s_adj = add_self_loop(source_s_adj) 
source_t_adj = add_self_loop(source_d_adj) 
source_od_adj = add_self_loop(source_od_adj) 
target_s_adj = add_self_loop(target_s_adj) 
target_t_adj = add_self_loop(target_d_adj) 
target_od_adj = add_self_loop(target_od_adj) 
print("Source graphs: ")
print("prox_adj: %d nodes, %d edges" % (source_prox_adj.shape[0], np.sum(source_prox_adj)))
print("road adj: %d nodes, %d edges" % (source_road_adj.shape[0], np.sum(source_road_adj > 0)))
print("poi_adj, %d nodes, %d edges" % (source_poi_adj.shape[0], np.sum(source_poi_adj > 0)))
print("s_adj, %d nodes, %d edges" % (source_s_adj.shape[0], np.sum(source_s_adj > 0)))
print("d_adj, %d nodes, %d edges" % (source_d_adj.shape[0], np.sum(source_d_adj > 0)))
print()
print("Target graphs:")
print("prox_adj: %d nodes, %d edges" % (target_prox_adj.shape[0], np.sum(target_prox_adj)))
print("road adj: %d nodes, %d edges" % (target_road_adj.shape[0], np.sum(target_road_adj > 0)))
print("poi_adj, %d nodes, %d edges" % (target_poi_adj.shape[0], np.sum(target_poi_adj > 0)))
print("s_adj, %d nodes, %d edges" % (target_s_adj.shape[0], np.sum(target_s_adj > 0)))
print("d_adj, %d nodes, %d edges" % (target_d_adj.shape[0], np.sum(target_d_adj > 0)))
print()
source_graphs = adjs_to_graphs([source_prox_adj, source_road_adj, source_poi_adj, source_s_adj, source_d_adj])
target_graphs = adjs_to_graphs([target_prox_adj, target_road_adj, target_poi_adj, target_s_adj, target_d_adj])
for i in range(len(source_graphs)):
    source_graphs[i] = source_graphs[i].to(device)
    target_graphs[i] = target_graphs[i].to(device)

# This function is a preparation for the edge type discriminator
def graphs_to_edge_labels(graphs):
    edge_label_dict = {}
    for i, graph in enumerate(graphs):
        src, dst = graph.edges()
        for s, d in zip(src, dst):
            s = s.item()
            d = d.item()
            if (s, d) not in edge_label_dict:
                edge_label_dict[(s, d)] = np.zeros(len(graphs))
            edge_label_dict[(s, d)][i] = 1
    edges = []
    edge_labels = [] 
    for k in edge_label_dict.keys():
        edges.append(k)
        edge_labels.append(edge_label_dict[k])
    edges = np.array(edges)
    edge_labels = np.array(edge_labels)
    return edges, edge_labels

source_edges, source_edge_labels = graphs_to_edge_labels(source_graphs)
target_edges, target_edge_labels = graphs_to_edge_labels(target_graphs)

# build models
# we need one embedding model, one scoring model, one prediction model
class MVGAT(nn.Module):
    def __init__(self, num_graphs=3, num_gat_layer=2, in_dim=14, hidden_dim=64, emb_dim=32, num_heads=2, residual=True):
        super().__init__()
        self.num_graphs = num_graphs 
        self.num_gat_layer = num_gat_layer 
        self.in_dim = in_dim 
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads 
        self.residual = residual 

        self.multi_gats = nn.ModuleList()
        for j in range(self.num_gat_layer):
            gats = nn.ModuleList() 
            for i in range(self.num_graphs):
                if j == 0:
                    gats.append(GATConv(self.in_dim, 
                                        self.hidden_dim, 
                                        self.num_heads, 
                                        residual=self.residual, 
                                        allow_zero_in_degree = True))
                elif j == self.num_gat_layer - 1:
                    gats.append(GATConv(self.hidden_dim * self.num_heads, 
                                        self.emb_dim // self.num_heads, 
                                        self.num_heads, 
                                        residual=self.residual, 
                                        allow_zero_in_degree = True))
                else:
                    gats.append(GATConv(self.hidden_dim * self.num_heads, 
                                        self.hidden_dim, 
                                        self.num_heads, 
                                        residual=self.residual, 
                                        allow_zero_in_degree=True))
            self.multi_gats.append(gats)
    
    def forward(self, graphs, feat):
        views = []
        for i in range(self.num_graphs):
            for j in range(self.num_gat_layer):
                if j == 0:
                    z = self.multi_gats[j][i](graphs[i], feat)
                else:
                    z = self.multi_gats[j][i](graphs[i], z)
                if j != self.num_gat_layer - 1:
                    z = F.relu(z)
                z = z.flatten(1)
            views.append(z)
        return views

class FusionModule(nn.Module):
    def __init__(self, num_graphs, emb_dim, alpha):
        super().__init__()
        self.num_graphs = num_graphs
        self.emb_dim = emb_dim
        self.alpha = alpha

        self.fusion_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.self_q = nn.ModuleList()
        self.self_k = nn.ModuleList()
        for i in range(self.num_graphs):
            self.self_q.append(nn.Linear(self.emb_dim, self.emb_dim))
            self.self_k.append(nn.Linear(self.emb_dim, self.emb_dim))

    def forward(self, views):        
        # run fusion by self attention
        cat_views = torch.stack(views, dim = 0)
        self_attentions = []
        for i in range(self.num_graphs):
            Q = self.self_q[i](cat_views)
            K = self.self_k[i](cat_views)
            # (3, num_nodes, 64)
            attn = F.softmax(torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.emb_dim), dim = -1)
            # (3, num_nodes, num_nodes)
            output = torch.matmul(attn, cat_views)
            self_attentions.append(output)
        self_attentions = sum(self_attentions) / self.num_graphs
        # (3, num_nodes, 64 * 2)
        for i in range(self.num_graphs):
            views[i] = self.alpha * self_attentions[i] + (1-self.alpha) * views[i]

        # further run multi-view fusion
        mv_outputs = []
        for i in range(self.num_graphs):
            mv_outputs.append(torch.sigmoid(self.fusion_linear(views[i])) * views[i])
        
        fused_outputs = sum(mv_outputs)
        # next_in = [(view + fused_outputs) / 2 for view in views]
        return fused_outputs, [(views[i] + fused_outputs) / 2 for i in range(self.num_graphs)]

class Scoring(nn.Module):
    def __init__(self, emb_dim, source_mask, target_mask): 
        super().__init__()
        self.emb_dim = emb_dim
        self.score = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim // 2), 
                                  nn.ReLU(inplace=True), 
                                  nn.Linear(self.emb_dim // 2, self.emb_dim // 2))
        self.source_mask = source_mask
        self.target_mask = target_mask
    
    def forward(self, source_emb, target_emb):
        target_context = torch.tanh(self.score(target_emb[self.target_mask.view(-1).bool()]).mean(0))
        source_trans_emb = self.score(source_emb)
        source_score = (source_trans_emb * target_context).sum(1)
        # the following lines modify inner product similarity to cosine similarity
        # target_norm = target_context.pow(2).sum().pow(1/2)
        # source_norm = source_trans_emb.pow(2).sum(1).pow(1/2)
        # source_score /= source_norm
        # source_score /= target_norm
        # print(source_score)
        return F.relu(torch.tanh(source_score))[self.source_mask.view(-1).bool()]

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
mmd = MMD_loss()

class EdgeTypeDiscriminator(nn.Module):
    def __init__(self, num_graphs, emb_dim):
        super().__init__()
        self.num_graphs = num_graphs
        self.emb_dim = emb_dim
        self.edge_network = nn.Sequential(nn.Linear(2 * self.emb_dim, self.emb_dim), 
                                          nn.ReLU(), 
                                          nn.Linear(self.emb_dim, self.num_graphs))
    def forward(self, src_embs, dst_embs):
        edge_vec = torch.cat([src_embs, dst_embs], dim = 1)
        return self.edge_network(edge_vec)

num_gat_layers = 2
in_dim = 14 
hidden_dim = 64
emb_dim = 64
num_heads = 2 
mmd_w = args.mmd_w
et_w = args.et_w
ma_param = args.ma_coef

mvgat = MVGAT(len(source_graphs), num_gat_layers, in_dim, hidden_dim, emb_dim, num_heads, True).to(device)
fusion = FusionModule(len(source_graphs), emb_dim, 0.8).to(device)
scoring = Scoring(emb_dim, th_mask_source, th_mask_target).to(device)
edge_disc = EdgeTypeDiscriminator(len(source_graphs), emb_dim).to(device)
mmd = MMD_loss()
# we still need a scoring model. 
# [NS, 64], [NT, 64] -> [NS]

# build model 
if args.model == 'STResNet':
    net = STResNet(len(lag), 1, 3).to(device)
elif args.model == 'STNet_nobn':
    net = STNet_nobn(1, 3, th_mask_target).to(device)
    print(net) 
elif args.model == 'STNet':
    net = STNet(1, 3, th_mask_target).to(device) 
    print(net) 

pred_optimizer = optim.Adam(net.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay) 
emb_param_list = list(mvgat.parameters()) + list(fusion.parameters()) + list(edge_disc.parameters())
emb_optimizer = optim.Adam(emb_param_list, lr = args.learning_rate, weight_decay = args.weight_decay) 
meta_optimizer = optim.Adam(scoring.parameters(), lr = args.outerlr, weight_decay = args.weight_decay) 
best_val_rmse = 999
best_test_rmse = 999 
best_test_mae = 999 

if args.save_name is not None:
    if not os.path.exists("../saved_models/%s/%s/%s/" % (args.scity, dataname, datatype)):
        os.makedirs("../saved_models/%s/%s/%s/" % (args.scity, dataname, datatype))

def evaluate(net_, loader, spatial_mask):
    net_.eval()
    with torch.no_grad():
        se = 0
        ae = 0
        valid_points = 0
        for it_ in loader:
            if len(it_) == 2:
                (x, y) = it_
            elif len(it_) == 4:
                _, _, x, y = it_
            x = x.to(device)
            y = y.to(device)
            lng = x.shape[2]
            lat = x.shape[3]
            out = net_(x, spatial_mask = spatial_mask.bool())
            valid_points += x.shape[0] * spatial_mask.sum().item()
            if len(out.shape) == 4: # STResNet
                se += (((out - y) ** 2) * (spatial_mask.view(1, 1, lng, lat))).sum().item()
                ae += ((out - y).abs() * (spatial_mask.view(1, 1, lng, lat))).sum().item()
            elif len(out.shape) == 3: # STNet
                batch_size = y.shape[0] 
                lag = y.shape[1]
                y = y.view(batch_size, lag, -1)[:, :, spatial_mask.view(-1).bool()]
                # print("out", out.shape)
                # print("y", y.shape)
                se += ((out - y) ** 2).sum().item()
                ae += (out - y).abs().sum().item()
    return np.sqrt(se / valid_points), ae / valid_points

def batch_sampler(tensor_list, batch_size):
    num_samples = tensor_list[0].size(0)
    idx = np.random.permutation(num_samples)[:batch_size]
    return (x[idx] for x in tensor_list)

def get_weights_bn_vars(module):
    fast_weights = OrderedDict(module.named_parameters())
    bn_vars = OrderedDict()
    for k in module.state_dict():
        if k not in fast_weights.keys():
            bn_vars[k] = module.state_dict()[k]
    return fast_weights, bn_vars

def train_epoch(net_, loader_, optimizer_, weights = None, mask = None, num_iters = None):
    net_.train()
    epoch_loss = []
    for i, (x, y) in enumerate(loader_):
        x = x.to(device)
        y = y.to(device)
        out = net_(x, spatial_mask = mask.bool())
        if len(out.shape) == 4: # STResNet
            eff_batch_size = y.shape[0]
            loss = ((out - y) ** 2).view(eff_batch_size, 1, -1)[:, :, mask.view(-1).bool()]
            # print("loss", loss.shape)
            if weights is not None:
                loss = (loss * weights)
                # print("weights", weights.shape)
                # print("loss * weights", loss.shape)
                loss = loss.mean(0).sum()
            else:
                loss = loss.mean(0).sum()
        elif len(out.shape) == 3: # STNet 
            eff_batch_size = y.shape[0] 
            y = y.view(eff_batch_size, 1, -1)[:, :, mask.view(-1).bool()] 
            loss = ((out - y) ** 2)
            if weights is not None:
                # print(loss.shape)
                # print(weights.shape)
                loss = (loss * weights.view(1, 1, -1)).mean(0).sum() 
            else:
                loss = loss.mean(0).sum()
        optimizer_.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_.parameters(), max_norm = 2)
        optimizer_.step()
        epoch_loss.append(loss.item()) 
        if num_iters is not None and num_iters == i:
            break
    return epoch_loss

def train_rt_epoch(net_, loader_, optimizer_):
    net_.train()
    epoch_predloss = []
    epoch_rtloss = []
    epoch_loss = []
    for i, (source_x, source_y, target_x, target_y) in enumerate(loader_):
        source_x = source_x.to(device)
        source_y = source_y.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)
        source_feat, _ = net_(source_x, spatial_mask = th_mask_source.bool(), return_feat = True)
        target_feat, target_out = net_(target_x, return_feat=True)
        batch_size = target_y.shape[0]
        lag = target_y.shape[1]
        target_y = target_y.view(batch_size, lag, -1)[:, :, th_mask_target.view(-1).bool()]
        loss_pred = ((target_out - target_y) ** 2).mean(0).sum()
        matching_source_feat = source_feat[:, matching_indices, :]
        loss_rt = (((target_feat - matching_source_feat) ** 2).sum(2) * matching_weight).sum(1).mean()
        loss = loss_pred + args.rt_weight * loss_rt
        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()
        epoch_predloss.append(loss_pred.item())
        epoch_rtloss.append(loss_rt.item())
        epoch_loss.append(loss.item())
    return np.mean(epoch_predloss), np.mean(epoch_rtloss), np.mean(epoch_loss)

def forward_emb(graphs_, in_feat_, od_adj_, poi_cos_):
    views = mvgat(graphs_, torch.Tensor(in_feat_).to(device))
    fused_emb, embs = fusion(views) 
    s_emb = embs[-2]
    d_emb = embs[-1]
    poi_emb = embs[-3] 
    recons_sd = torch.matmul(s_emb, d_emb.transpose(0, 1))
    pred_d = torch.log(torch.softmax(recons_sd, dim = 1) + 1e-5) 
    loss_d = (torch.Tensor(od_adj_).to(device) * pred_d).mean() 
    pred_s = torch.log(torch.softmax(recons_sd, dim = 0) + 1e-5) 
    loss_s = (torch.Tensor(od_adj_).to(device) * pred_s).mean() 
    poi_sim = torch.matmul(poi_emb, poi_emb.transpose(0, 1)) 
    loss_poi = ((poi_sim - torch.Tensor(poi_cos_).to(device)) ** 2).mean()
    loss = -loss_s - loss_d + loss_poi 

    return loss, fused_emb, embs

    

def meta_train_epoch(s_embs, t_embs):
    meta_query_losses = []
    for meta_ep in range(args.outeriter):
        fast_losses = []
        fast_weights, bn_vars = get_weights_bn_vars(net) 
        source_weights = scoring(s_embs, t_embs)
        # inner loop on source, pre-train with weights
        for meta_it in range(args.sinneriter):
            s_x, s_y = batch_sampler((torch.Tensor(source_x), torch.Tensor(source_y)), args.batch_size) 
            s_x = s_x.to(device) 
            s_y = s_y.to(device) 
            pred_source = net.functional_forward(s_x, th_mask_source.bool(), fast_weights, bn_vars, bn_training=True) 
            if len(pred_source.shape) == 4: # STResNet 
                loss_source = ((pred_source - s_y) ** 2).view(args.batch_size, 1, -1)[:, :, th_mask_source.view(-1).bool()]
                # print(loss_source.shape)
                loss_source = (loss_source * source_weights).mean(0).sum() 
            elif len(pred_source.shape) == 3: # STNet 
                s_y = s_y.view(args.batch_size, 1, -1)[:, :, th_mask_source.view(-1).bool()] 
                loss_source = (((pred_source - s_y) ** 2) * source_weights.view(1, 1, -1))
                # print(loss_source.shape)
                # print(source_weights.shape)
                loss_source = loss_source.mean(0).sum()
            fast_loss = loss_source 
            fast_losses.append(fast_loss.item()) #
            grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True) 
            for name, grad in zip(fast_weights.keys(), grads):
                fast_weights[name] = fast_weights[name] - args.innerlr * grad 
                # fast_weights[name].add_(grad, alpha = -args.innerlr)

        # inner loop on target, simulate fine-tune
        
        for meta_it in range(args.tinneriter):
            t_x, t_y = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size) 
            t_x = t_x.to(device) 
            t_y = t_y.to(device) 
            pred_t = net.functional_forward(t_x, th_mask_target.bool(), fast_weights, bn_vars, bn_training=True) 
            if len(pred_t.shape) == 4: # STResNet 
                loss_t = ((pred_t - t_y) ** 2).view(args.batch_size, 1, -1)[:, :, th_mask_target.view(-1).bool()]
                # print(loss_source.shape)
                loss_t = loss_t.mean(0).sum()
            elif len(pred_t.shape) == 3: # STNet 
                t_y = t_y.view(args.batch_size, 1, -1)[:, :, th_mask_target.view(-1).bool()] 
                # print(t_y.shape)
                loss_t = ((pred_t - t_y) ** 2)# .view(1, 1, -1))
                # print(loss_t.shape)
                # print(loss_source.shape)
                # print(source_weights.shape)
                loss_t = loss_t.mean(0).sum()
            fast_loss = loss_t
            fast_losses.append(fast_loss.item()) #
            grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True) 
            for name, grad in zip(fast_weights.keys(), grads):
                fast_weights[name] = fast_weights[name] - args.innerlr * grad 
                # fast_weights[name].add_(grad, alpha = -args.innerlr)
        
        q_losses = []
        target_iter = max(args.sinneriter, args.tinneriter)
        for k in range(3):
            # query loss 
            x_q,  y_q = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size) 
            x_q = x_q.to(device) 
            y_q = y_q.to(device) 
            pred_q = net.functional_forward(x_q, th_mask_target.bool(), fast_weights, bn_vars, bn_training = True)
            if len(pred_q.shape) == 4: # STResNet 
                loss = (((pred_q - y_q) ** 2) * (th_mask_target.view(1, 1, lng_target, lat_target)))
                loss = loss.mean(0).sum()
            elif len(pred_q.shape) == 3: # STNet
                y_q = y_q.view(args.batch_size, 1, -1)[:, :, th_mask_target.view(-1).bool()]
                loss = ((pred_q - y_q) ** 2).mean(0).sum()
            q_losses.append(loss) 
        q_loss = torch.stack(q_losses).mean() 
        weights_mean = (source_weights**2).mean()
        meta_loss = q_loss + weights_mean * args.weight_reg
        meta_optimizer.zero_grad() 
        meta_loss.backward(inputs = list(scoring.parameters()))
        torch.nn.utils.clip_grad_norm_(scoring.parameters(), max_norm = 2)
        meta_optimizer.step()
        meta_query_losses.append(q_loss.item()) 
    return np.mean(meta_query_losses)

def train_emb_epoch():
    loss_source, fused_emb_s, embs_s = forward_emb(source_graphs, source_norm_poi, source_od_adj, source_poi_cos)
    loss_target, fused_emb_t, embs_t = forward_emb(target_graphs, target_norm_poi, target_od_adj, target_poi_cos)
    loss_emb = loss_source+loss_target 
    # compute domain adaptation loss
    source_ids = np.random.randint(0, np.sum(mask_source), size = (128, ))
    target_ids = np.random.randint(0, np.sum(mask_target), size = (128, ))
    mmd_loss = mmd(fused_emb_s[th_mask_source.view(-1).bool()][source_ids, :], fused_emb_t[th_mask_target.view(-1).bool()][target_ids, :])

    source_batch_edges = np.random.randint(0, len(source_edges), size = (256, ))
    target_batch_edges = np.random.randint(0, len(target_edges), size = (256, ))
    source_batch_src = torch.Tensor(source_edges[source_batch_edges, 0]).long()
    source_batch_dst = torch.Tensor(source_edges[source_batch_edges, 1]).long()
    source_emb_src = fused_emb_s[source_batch_src, :]
    source_emb_dst = fused_emb_s[source_batch_dst, :]
    target_batch_src = torch.Tensor(target_edges[target_batch_edges, 0]).long()
    target_batch_dst = torch.Tensor(target_edges[target_batch_edges, 1]).long()
    target_emb_src = fused_emb_t[target_batch_src, :]
    target_emb_dst = fused_emb_t[target_batch_dst, :]   

    pred_source = edge_disc.forward(source_emb_src, source_emb_dst)
    pred_target = edge_disc.forward(target_emb_src, target_emb_dst)
    source_batch_labels = torch.Tensor(source_edge_labels[source_batch_edges]).to(device)
    target_batch_labels = torch.Tensor(target_edge_labels[target_batch_edges]).to(device)
    loss_et_source = -((source_batch_labels * torch.log(torch.sigmoid(pred_source) + 1e-6)) + (1 - source_batch_labels) * torch.log(1 - torch.sigmoid(pred_source) + 1e-6)).sum(1).mean()
    loss_et_target = -((target_batch_labels * torch.log(torch.sigmoid(pred_target) + 1e-6)) + (1 - target_batch_labels) * torch.log(1 - torch.sigmoid(pred_target) + 1e-6)).sum(1).mean()
    loss_et = loss_et_source + loss_et_target

    emb_optimizer.zero_grad()
    loss = loss_emb + mmd_w * mmd_loss + et_w * loss_et
    loss.backward()
    emb_optimizer.step()
    return loss_emb.item(), mmd_loss.item(), loss_et.item()

emb_losses = []
mmd_losses = []
edge_losses = []
pretrain_emb_epoch = 80
for emb_ep in range(pretrain_emb_epoch):
    loss_emb_, loss_mmd_, loss_et_ = train_emb_epoch()
    emb_losses.append(loss_emb_)
    mmd_losses.append(loss_mmd_)
    edge_losses.append(loss_et_)
print("[%.2fs]Pretrain embeddings for %d epochs, average emb loss %.4f, mmd loss %.4f, edge loss %.4f" % (time.time() - start_time, pretrain_emb_epoch, np.mean(emb_losses), np.mean(mmd_losses), np.mean(edge_losses)))
with torch.no_grad():
    views = mvgat(source_graphs, torch.Tensor(source_norm_poi).to(device))
    fused_emb_s, _ = fusion(views) 
    views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device)) 
    fused_emb_t, _ = fusion(views) 
emb_s = fused_emb_s.cpu().numpy()[mask_source.reshape(-1)]
emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)] 
logreg = LogisticRegression(max_iter=500)
cvscore_s = cross_validate(logreg, emb_s, source_emb_label)['test_score'].mean()
cvscore_t = cross_validate(logreg, emb_t, target_emb_label)['test_score'].mean()
print("[%.2fs]Pretraining embedding, source cvscore %.4f, target cvscore %.4f" % \
    (time.time() - start_time, cvscore_s, cvscore_t))  
print()

source_weights_log = []

for ep in range(num_epochs):
    net.train()
    mvgat.train()
    fusion.train()
    scoring.train()

    # train embeddings
    emb_losses = []
    mmd_losses = []
    edge_losses = []
    for emb_ep in range(5):
        loss_emb_, loss_mmd_, loss_et_ = train_emb_epoch()
        emb_losses.append(loss_emb_)
        mmd_losses.append(loss_mmd_)
        edge_losses.append(loss_et_)
    # evaluate embeddings 
    with torch.no_grad():
        views = mvgat(source_graphs, torch.Tensor(source_norm_poi).to(device))
        fused_emb_s, _ = fusion(views) 
        views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device)) 
        fused_emb_t, _ = fusion(views) 
    if ep % 2 == 0:
        emb_s = fused_emb_s.cpu().numpy()[mask_source.reshape(-1)]
        emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)] 
        mix_embs = np.concatenate([emb_s, emb_t], axis = 0)
        mix_labels = np.concatenate([source_emb_label, target_emb_label])
        logreg = LogisticRegression(max_iter=500)
        cvscore_s = cross_validate(logreg, emb_s, source_emb_label)['test_score'].mean()
        cvscore_t = cross_validate(logreg, emb_t, target_emb_label)['test_score'].mean()
        cvscore_mix = cross_validate(logreg, mix_embs, mix_labels)['test_score'].mean()
        print("[%.2fs]Epoch %d, embedding loss %.4f, mmd loss %.4f, edge loss %.4f, source cvscore %.4f, target cvscore %.4f, mixcvscore %.4f" % \
            (time.time() - start_time, ep, np.mean(emb_losses), np.mean(mmd_losses), np.mean(edge_losses), cvscore_s, cvscore_t, cvscore_mix))    
    if ep == num_epochs - 1:
        emb_s = fused_emb_s.cpu().numpy()[mask_source.reshape(-1)]
        emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)] 
        # np.save("%s.npy" % args.scity, arr = emb_s)
        # np.save("%s.npy" % args.tcity, arr = emb_t)
        with torch.no_grad():
            trans_emb_s = scoring.score(fused_emb_s)
            trans_emb_t = scoring.score(fused_emb_t)
        # np.save("%s_trans.npy" % args.scity, arr = trans_emb_s.cpu().numpy()[mask_source.reshape(-1)])
        # np.save("%s_trans.npy" % args.tcity, arr = trans_emb_t.cpu().numpy()[mask_target.reshape(-1)])
        
    
    # meta train scorings
    avg_q_loss = meta_train_epoch(fused_emb_s, fused_emb_t) 
    with torch.no_grad(): 
        source_weights = scoring(fused_emb_s, fused_emb_t)
    
    # For debug: use fixed weightings.
    # with torch.no_grad(): 
    #     source_weights_ = scoring(fused_emb_s, fused_emb_t)
    # avg_q_loss = 0
    # source_weights = torch.ones_like(source_weights_)
    
    # implement a moving average
    if ep == 0:
        source_weights_ma = torch.ones_like(source_weights, device = device, requires_grad=False)
    source_weights_ma = ma_param * source_weights_ma + (1 - ma_param) * source_weights
    source_weights_log.append(source_weights.cpu().numpy())
    # train network on source
    source_loss = train_epoch(net, source_loader, pred_optimizer, weights = source_weights_ma, mask = th_mask_source, num_iters = args.pretrain_iter)
    avg_source_loss = np.mean(source_loss)
    avg_target_loss = evaluate(net, target_train_loader, spatial_mask = th_mask_target)[0]
    print("[%.2fs]Epoch %d, average meta query loss %.4f, source weight mean %.4f, var %.6f, source loss %.4f, target_loss %.4f" % \
        (time.time() - start_time, ep, avg_q_loss, source_weights_ma.mean().item(), torch.var(source_weights_ma).item(), avg_source_loss, avg_target_loss))
    print(torch.var(source_weights).item())
    print(source_weights.mean().item())
    if source_weights_ma.mean() < 0.005:
        # stop pre-training
        break
    net.eval()
    rmse_val, mae_val = evaluate(net, target_val_loader, spatial_mask = th_mask_target)
    rmse_s_val, mae_s_val = evaluate(net, source_loader, spatial_mask = th_mask_source)
    print("Epoch %d, source validation rmse %.4f, mae %.4f" % (ep, rmse_s_val * (smax - smin), mae_s_val * (smax - smin)))
    print("Epoch %d, target validation rmse %.4f, mae %.4f" % (ep, rmse_val * (max_val - min_val), mae_val * (max_val - min_val)))
    print()

if args.save_name is not None:
    torch.save(net.state_dict(), '../saved_models/%s/%s/%s/%s_%s.pt' % (args.scity, dataname, datatype, args.model, args.save_name))
log_info = []
for ep in range(num_epochs, 80 + num_epochs):
    # fine-tuning 
    net.train() 
    avg_predloss, avg_rtloss, avg_loss = train_rt_epoch(net, target_train_loader, pred_optimizer)
    print('[%.2fs]Epoch %d, target pred loss %.4f, rt loss %.4f' % (time.time() - start_time, ep, avg_predloss, avg_rtloss))
    net.eval() 
    rmse_val, mae_val = evaluate(net, target_val_loader, spatial_mask = th_mask_target)
    rmse_test, mae_test = evaluate(net, target_test_loader, spatial_mask = th_mask_target)
    if rmse_val < best_val_rmse:
        best_val_rmse = rmse_val 
        best_test_rmse = rmse_test 
        best_test_mae = mae_test
        print("Update best test...")
    print("validation rmse %.4f, mae %.4f" % (rmse_val * (max_val - min_val), mae_val * (max_val - min_val)))
    print("test rmse %.4f, mae %.4f" % (rmse_test * (max_val - min_val), mae_test * (max_val - min_val)))
    print()
    log_info_dict = {
        "train_loss": avg_target_loss, 
        "rmse_val": rmse_val * (max_val - min_val), 
        "mae_val": mae_val * (max_val - min_val), 
        "rmse_test": rmse_test * (max_val - min_val), 
        "mae_test": mae_test * (max_val - min_val)
    }
    log_info.append(log_info_dict)   

print("Best test rmse %.4f, mae %.4f" % (best_test_rmse * (max_val - min_val), best_test_mae * (max_val - min_val)))
if args.log_name is not None:
    log_folder_path = "../log/%s_%s_%s/" % (args.scity, args.tcity, args.dataname)
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    with open(log_folder_path + "%s_%d_%s" % (args.datatype, args.data_amount, args.log_name), 'w') as infile:
        infile.write(json.dumps(log_info, indent=4))
    weight_path = log_folder_path +'weights_%d_%s_%s/' % (args.data_amount, args.datatype, args.log_name)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    for i in range(num_epochs):
        np.save(weight_path+'%d.npy' % i, arr = source_weights_log[i])
