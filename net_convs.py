
from torch import nn
import torch
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from torch_scatter import scatter 

class Conv(nn.Module):
    def __init__(self, irreps_in_src, irreps_in_dst, irreps_sh, edge_dim, irreps_out, hidden_dim, edge_key = None, safety_checks = False):
        super().__init__()

        self.edge_key = edge_key
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in_src,            # sender
            irreps_in2=irreps_sh,            # spherical harmonics
            irreps_out=irreps_out,           # message output irreps
            shared_weights = False
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.tp.weight_numel)
        )
        self.irreps_in_src = irreps_in_src
        self.irreps_in_dst = irreps_in_dst
        self.irreps_out = irreps_out
        self.irreps_sh = irreps_sh
        self.residual_proj = Linear(irreps_in_dst, irreps_out)

        self.safety_checks = False 

    def forward(self,dst, x_src, x_dst, sh, edge_attr,reduce ='mean', num_graphs = None):
        w = self.edge_mlp(edge_attr)
        msg = self.tp(x_src, sh, w)  
        msg = scatter(msg, dst, dim=0, dim_size=num_graphs if x_dst==None else x_dst.shape[0], reduce=reduce)
        if x_dst != None: 
            msg = self.residual_proj(x_dst) + msg  

        if self.safety_checks: 
            self.check_nan(msg)
        
        return msg
 
    
    def check_nan(self, msg): 
            print(f"Conv: irrep in src/dst and out {self.irreps_in_src},{self.irreps_in_dst},{self.irreps_out}")
            print("  has_nan:", torch.isnan(msg).any().item())
            print("  has_inf:", torch.isinf(msg).any().item())
            print("  min:", msg.min().item())
            print("  max:", msg.max().item())
            print("  msg.dtype:", msg.dtype)
            print("  msg.device:", msg.device)
