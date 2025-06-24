
from torch import nn
import torch
from e3nn.o3 import FullyConnectedTensorProduct, Linear, Irreps 
from torch_scatter import scatter 
from e3nn.nn import Gate, Activation 

class Conv(nn.Module):
    def __init__(self, irreps_in_src, irreps_in_dst, irreps_sh, edge_dim, irreps_out, hidden_dim, edge_key = None, safety_checks = False):
        super().__init__()

        self.edge_key = edge_key
 
        # gate https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
        # scalars: 0e part of tp output, split with gates
        # gates: 0e part of tp output, split with scalars. same dim as gated, nonlinearity applied 
        # gated: 0o, 1e, 1o ,... part of tp output. 
        # gate takes a concatenation of these three (new irreps_out) and outputs the original irreps_out. 
        irreps_scalars, irreps_gates, irreps_gated = split_for_gate(irreps_out)
        
        if irreps_gated is not None: 
            self.nl = Gate(
                irreps_scalars,
                [nn.ReLU()]*len(irreps_scalars),      
                irreps_gates,
                [torch.sigmoid]*len(irreps_gates),
                irreps_gated)
            # print(f"sanity check {self.nl.irreps_in == irreps_scalars + irreps_gates + irreps_gated}")
            irreps_out = str(irreps_scalars + irreps_gates + irreps_gated)
        else: 
            self.nl = Activation(irreps_out,[nn.ReLU()])
       

    

        self.tp = FullyConnectedTensorProduct(
                irreps_in1=irreps_in_src,            # sender
                irreps_in2=irreps_sh,            # spherical harmonics
                irreps_out=irreps_out,           # message output irreps
                shared_weights = False
            )


        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(), # tanh
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

        msg = self.nl(msg)
        return msg
 
    
    def check_nan(self, msg): 
            print(f"Conv: irrep in src/dst and out {self.irreps_in_src},{self.irreps_in_dst},{self.irreps_out}")
            print("  has_nan:", torch.isnan(msg).any().item())
            print("  has_inf:", torch.isinf(msg).any().item())
            print("  min:", msg.min().item())
            print("  max:", msg.max().item())
            print("  msg.dtype:", msg.dtype)
            print("  msg.device:", msg.device)

def split_for_gate(irreps_out: Irreps):
    irreps_out = Irreps(irreps_out).simplify()

    # Separate scalar irreps (l=0)
    scalar_list = []
    gated_list = []
    mul_scalars = 0 

    for mul, ir in irreps_out:
        if ir.l == 0 and ir.p == 1:  # keep only 0e
            scalar_list.append((mul, ir))
            mul_scalars = mul
        else:
            gated_list.append((mul, ir))


    irreps_scalars = Irreps(scalar_list)
    irreps_gated   = Irreps(gated_list) if len(gated_list)>0 else None 
    irreps_gates   = Irreps(f"{irreps_gated.num_irreps}x0e") if len(gated_list)>0 else None

    # print("scalars",irreps_scalars)
    # print("gated",irreps_gated)
    # print("gates",irreps_gates)

    return irreps_scalars, irreps_gates, irreps_gated

