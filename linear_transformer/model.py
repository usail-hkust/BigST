import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m/d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)

def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)

def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    return data_dash

def numerator(qs, ks, vs):
    kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs) # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)

def denominator(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)

def kernelized_softmax(x, query, key):
    # x: [B, N, H, D] query: [B, N, H, m], key: [B, N, H, m]
    query = query.permute(1, 0, 2, 3) # [N, B, H, m]
    key = key.permute(1, 0, 2, 3) # [N, B, H, m]
    x = x.permute(1, 0, 2, 3) # [N, B, H, D]

    z_num = numerator(query, key, x) # [N, B, H, D]
    z_den = denominator(query, key) # [N, H]

    z_num = z_num.permute(1, 0, 2, 3)  # [B, N, H, D]
    z_den = z_den.permute(1, 0, 2)
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = z_num / z_den # # [B, N, H, D]

    return z_output

class kernelized_attention(nn.Module):
    def __init__(self, c_in, c_out, dropout, random_feature_dim=30, tau=1.0, num_heads=4):
        super(kernelized_attention, self).__init__()
        self.Wk = nn.Linear(c_in, c_out * num_heads)
        self.Wq = nn.Linear(c_in, c_out * num_heads)
        self.Wv = nn.Linear(c_in, c_out * num_heads)
        self.Wo = nn.Linear(c_out * num_heads, c_out)
        self.c_in = c_in
        self.c_out = c_out
        self.num_heads = num_heads
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.activation = nn.ReLU
        self.dropout = dropout
        
    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()

    def forward(self, x):
        B, T = x.size(0), x.size(1) # (B, T, D)
        query = self.Wq(x).reshape(-1, T, self.num_heads, self.c_out) # (B, T, H, D)
        key = self.Wk(x).reshape(-1, T, self.num_heads, self.c_out) # (B, T, H, D)
        x = self.Wv(x).reshape(-1, T, self.num_heads, self.c_out) # (B, T, H, D)
        
        dim = query.shape[-1] # (B, T, H, D)
        seed = torch.ceil(torch.abs(torch.sum(query) * 1e8)).to(torch.int32)
        projection_matrix = create_projection_matrix(self.random_feature_dim, dim, seed=seed).to(query.device) # (d, m)
        query = query / math.sqrt(self.tau)
        key = key / math.sqrt(self.tau)
        query = softmax_kernel_transformation(query, True, projection_matrix) # [B, T, H, m]
        key = softmax_kernel_transformation(key, False, projection_matrix) # [B, T, H, m]
        
        x = kernelized_softmax(x, query, key)
        
        x = self.Wo(x.flatten(-2, -1)) # (B, T, D)
        
        return x

class kernel_transformer(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3):
        super(kernel_transformer, self).__init__()
        self.tau = 1.0
        self.random_feature_dim = 48
        
        self.use_residual = True
        self.use_bn = False
        self.use_act = True
        
        self.dropout = dropout
        self.activation = nn.ReLU()
        
        self.fc_convs = nn.ModuleList()
        self.transformer_layer = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.filter_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 12), dilation=1)
        self.gate_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 12), dilation=1)
        
        self.position_embedding = nn.Parameter(torch.empty(288, 32), requires_grad=True)
        
        for i in range(3):
            self.transformer_layer.append(kernelized_attention(32, 32, self.dropout, self.random_feature_dim, self.tau))
            self.bn.append(nn.LayerNorm(32))
        
        self.regression_layer = nn.Linear(32, 12)

    def forward(self, x):
        # input: (1, 9638, 2016, 3) (B, N, T, D)
        B, N, T, D = x.size()
        pe = self.position_embedding[(x[:, :, :, 1]*288).type(torch.LongTensor)] # (B, N, T, 32)
        
        x = x.permute(0, 3, 1, 2) # (B, N, T, D) -> (B, D, N, T)
        x = nn.functional.pad(x, (11, 0, 0, 0))

        # convolution layer
        # select_list = [i+11 for i in range(0, 2016, 12)]
        x = self.filter_conv(x) # (B, D, N, T)
        # x = x[: ,:, :, select_list]

        # positional embedding layer
        B, D, N, T = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B*N, T, D) # (B, D, N, T) -> (B, N, T, D) -> (B*N, T, D)
        pe = pe.reshape(B*N, T, D)
        x = x+pe # (B*N, T, D)

        # kernelized attention
        for j in range(3):
            residual = x # (B*N, T, D)
            x = self.transformer_layer[j](x) # (B*N, T, D)
            x = self.bn[j](x)
            x = x+residual # (B*N, T, D)

        x = self.activation(x) # (B*N, T, D)
        x = torch.sum(x[:, -12:, :], dim=1) # (B*N, D)
        x = x.view(B, N, -1) # (B, N, D)
        x = self.regression_layer(x) # (B, N, T)
        return x
