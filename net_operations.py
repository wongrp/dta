

""" 
classes for 
1. encoding nodes 
2. smoothing distances 
3. encoding edges
"""

import torch 
from torch import nn
from e3nn import o3 


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()

        # Define learnable radius, using log to ensure positivity
        log_stop = torch.nn.Parameter(torch.log(torch.tensor(stop)))
        stop = torch.exp(log_stop)

        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EncodeAtom(torch.nn.Module):
    def __init__(self, num_atom_type,
                 hyb_types_num, hyb_embed_dim,
                 binary_features_num, binary_embed_dim,
                 scalar_features_num,out_dim):
        super().__init__()
        """
        atom_types_num: number of atomic element categories
        atom_embed_dim: length of atom type embedding vector 
        hyb_types_num: range of hybridization type (3)
        hyb_emb_dim: length of hybridization embedding vector
        binary_features_num: number of features that are either 0 or 1 
        binary_embed_dim: length of binary feature vector 
        scalar_featuers_num:  
        """ 
        self.atom_onehot_index = range(7)
        self.atom_binary_index = [7,8,13,14,15,16,17]
        self.atom_hyb_index = 9
        self.atom_scalar_index = [10,11,12]

        # Embeddings
        self.hyb_embedding = nn.Embedding(hyb_types_num, hyb_embed_dim)
        self.binary_embedding = nn.Embedding(2, binary_embed_dim)  # 0 or 1 input
        
        # Scalar features will be normalized outside before input
        self.scalar_features_num = scalar_features_num
        self.input_dim = num_atom_type+hyb_embed_dim+binary_embed_dim*binary_features_num +scalar_features_num
        self.linear=nn.Linear(self.input_dim, out_dim)
        # print(f"residue out dim {out_dim}")
        
        
    def forward(self, ft):
        """
        Inputs:
            atom_types: (N,) LongTensor
            hyb_types: (N,) LongTensor corresponding to hybridization 
            binary_flags: (N, num_binary_features) LongTensor (0/1)
            scalar_features: (N, num_scalar_features) FloatTensor
        
        Output:
            node_features: (N, total_feature_dim) FloatTensor
        """


        atom_types=ft[:,self.atom_onehot_index]
        hyb_types=ft[:,self.atom_hyb_index]
        binary_flags=ft[:,self.atom_binary_index]
        scalar_features=ft[:,self.atom_scalar_index]

        # Embed categorical
        # print(hyb_types)
        hyb_types = hyb_types.long()
        # print(hyb_types.unique())
        hyb_embedded = self.hyb_embedding(hyb_types)          # (N, hyb_embed_dim)
        # Each binary feature separately embedded then concatenated
        binary_embedded = self.binary_embedding(binary_flags.long())   # (N, num_binary_features, binary_embed_dim)
        binary_embedded = binary_embedded.view(binary_embedded.shape[0], -1)  # flatten last two dims
        
        # Scalar features passed directly
        scalar_normalized = scalar_features.float()  # assume already normalized outside

        # Concatenate everything
        node_features = self.linear(torch.cat([atom_types.float(), hyb_embedded, binary_embedded, scalar_normalized], dim=-1))
        # print(atom_types.shape)
        # print(hyb_embedded.shape)
        # print(binary_embedded.shape)
        # print(scalar_normalized.shape)
    
        # print(node_features.shape)
        # print(self.input_dim)


        return node_features


class EncodeResidueNodes(torch.nn.Module):
    def __init__(self, dimlist, ns):
        super().__init__()
        self.dimlist = dimlist
        print(f"residue dimensions {dimlist}")
        self.encode_cont = torch.nn.Linear(dimlist[0],ns) # mix continuous scalars
        self.encode_cat = torch.nn.Embedding(dimlist[2],ns) # categorical embedding 
        self.mix_scalars = torch.nn.Linear(ns+ns,ns) # scalar embedding 

    def forward(self,features): 
        print(f"residue features shape: {features.shape}")
        # vectors associated with residue --> reshape into nx3 
        continuous_scalars = self.encode_cont(features[:,:self.dimlist[0]]) # scalars 
        vectors = torch.reshape(features[:,self.dimlist[0]:self.dimlist[0]+self.dimlist[1]],(-1,int(self.dimlist[1]/3),3))
        # this step is because we haven't fixed the orientation vector. 
        # remove the second backbone orientation vector 

        # vectors have dims (B,N,3). If fed through linear layer, will mix each vector. instead, take transpose ...
        vectors = o3.spherical_harmonics(1, vectors, normalize=True, normalization="component") # after this should still be (B,N,3)
        vectors = vectors.flatten(start_dim = 1 ) # flatten to get (B,3N) for e3nn
        print(f"vectors dimension: {vectors.shape}")
        sequence = features[:,-1].long()# sequence is in last place. 
        categories = self.encode_cat(sequence) # categorical 
        scalars = self.mix_scalars(torch.cat([continuous_scalars, categories],dim = -1))
        encoding = torch.cat([scalars,vectors,],dim = -1) # dimensions of ns+ns+nv 
        print(f"encoding shape {encoding.shape}")
        return encoding 


class EncodeCombined(torch.nn.Module): 
    def __init__(self, num_atom_type,
                 hyb_types_num, hyb_embed_dim,
                 binary_features_num, binary_embed_dim,
                 scalar_features_num,atom_dim_list, ns):
        super().__init__()
        self.res_encoder = EncodeResidueNodes(atom_dim_list, ns)
        self.atom_encoder = EncodeAtom(num_atom_type,
                    hyb_types_num, hyb_embed_dim,
                    binary_features_num, binary_embed_dim,
                    scalar_features_num, ns)

    
    def forward(self,ft,split): 
        pre = ft.clone()
        mask1 = split
        mask2 = ~split
       
        out1 = self.res_encoder(pre[mask1])
        out2 = self.atom_encoder(pre[mask2])
        

        # strategy 1: pad the smaller tensor 
        final_dim = max(out1.shape[1], out2.shape[1])
        print(final_dim)
        pad1 = final_dim - out1.shape[1]
        pad2 = final_dim - out2.shape[1]

        if pad1 > 0:
                out1 = torch.cat([out1, torch.zeros(out1.size(0), pad1, device=ft.device)], dim=1)
        if pad2 > 0:
                out2 = torch.cat([out2, torch.zeros(out2.size(0), pad2, device=ft.device)], dim=1)
        
        return torch.cat([out1,out2],dim=0)
        