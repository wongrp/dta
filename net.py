from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear
from e3nn.nn import Gate

import torch.nn as nn
from itertools import chain 


import torch
from torch import nn
from net_operations import EncodeAtom, EncodeResidueNodes, EncodeCombined
from net_fusion import MLP, CrossAttention
from net_convs import Conv 

torch.set_printoptions(
    threshold=100,  
    linewidth=1000,    
    edgeitems=10     
) 
                       
class Net(nn.Module):
    def __init__(self, args, field,schedule,l_schedule,edge_sets):
        super(Net, self).__init__()

        self.alias_map = {
            'pp': 'pocket-pocket',
            'll': 'ligand-ligand',
            'rr': 'protein-protein',
            'cc': 'protein_ligand-protein_ligand',

            'pv': 'pocket-virtual_p',
            'vp': 'virtual_p-pocket',
            'lv': 'ligand-virtual_l',
            'vl': 'virtual_l-ligand',
            'rv': 'protein-virtual_r',
            'vr': 'virtual_r-protein',
            'rl': 'protein-ligand',
            'lr': 'ligand-protein',
            'cv': 'complex-virtual_c',
            'vc': 'virtual_c-complex',
            'xv': 'protein_ligand-virtual_c',
            'vx': 'virtual_c-protein_ligand',
            'zv': 'protein_pocket-virtual_z',
            'vz': 'virtual_z-protein_pocket',

            'p': 'pocket',
            'l': 'ligand', 
            'r': 'protein', 
            'c': 'pocket_ligand',
            'z': 'protein_pocket',
            'x': 'protein_ligand',

            'readout': {'c': 'virtual_c', 'x': 'virtual_x', 'z': 'virtual_z', 'r': 'virtual_r', 'l': 'virtual_l', 'p': 'virtual_p'}
        }

        # extract parameters
        self.inspect = args.inspect_tensors
        self.ns = args.ns  
        self.nv = args.nv 
        self.lsh = Irreps.spherical_harmonics(lmax=args.lsh)  
        self.edgedim = self.ns #3*
        self.hdim = self.ns #3*
        self.layer_outputs = [] 
        self.irrep_schedule = []

        # embedding dimensions 
        self.edge_in_dim = 32 # this is the hard-coded smoothed_dim in net_tranform_graphs.py. 
        self.atom_feature_dims = [7,7,1,3] 
        self.p_node_dim = [6,9,21] #scalar (dihedrals), vecs (orientations), then categorical 
        self.hyb_types_num = 7# 0 to 6. In reality there's only 1-6 so maybe 6? 
        self.hyb_embed_dim = 7
        self.binary_embed_dim = 2 
        self.bond_edge_dim = 12 
        self.res_node_dim = [6,9,21] #scalar (dihedrals), vecs (orientations), then categorical 

        # structures and interactions 
        self.field = field.copy()
        self.field_shorthand = self.field.copy()
        self.edge_field = set([item for layer in schedule for item in layer])
        self.schedule = schedule 
        self.expand_schedule() 
        self.schedule_shorthand = schedule 
        self.l_schedule = l_schedule 
        self.readout_dim = args.readout_dim

        # edge sets 
        self.edge_sets = edge_sets 

        # encoders 
        self.node_encoders = self.get_node_encoders()
        self.edge_encoders = self.get_edge_encoders(args.ns, args.dropout_edge)

        
        # get a list of convolutions and edge keys, and define fusing operations 
        self.get_convs()
        self.fuse = self.get_fuse(type=args.fusion_type,in_dim = self.readout_dim,out_dim = 1, num_objects=len(self.field_shorthand)) 


    def forward(self,data): 
        self.encode(data) 
        self.layer_outputs.clear()
        self.irrep_schedule.clear()   
        # print(f"check {data['protein', 'global', 'virtual_r'].edge_index[1].unique()}") 
        # print(f"check {data['pocket', 'global', 'virtual_p'].edge_index[1].unique()}") 
    
        for idx, conv in enumerate(self.convs):
            key = self.edge_keys[idx] 
            # print(f"conv key {key}")
            # print(f"CONV LAYER {idx+1} out of {len(self.convs)} for {key}")
            d = data[key]
            src, dst = d.edge_index 
            edge_attr = d.edge_attr
            sh = d.sh
            xsrc = data[key[0]].ft[src]
            xdst = data[key[2]].ft
            data[key[2]].ft = conv(dst,xsrc,xdst,sh,edge_attr) 
            if self.inspect: 
                # print(f"{key[2]} layer {idx} features {data[key[2]].ft.shape} \n: {data[key[2]].ft}")
                self.irrep_schedule.append({'name': f"hidden layer {idx} {key[2]}", 'irreps': [conv.irreps_in_src, conv.irreps_sh, conv.irreps_out,conv.irreps_in_dst]})
                self.layer_outputs.append({'name': f"hidden layer {idx} {key[2]}", 'ft': data[key[2]].ft.clone() })


        node_readouts = [] 
        # print(f"data is {data}")
        for idx, conv in enumerate(self.readouts): 
            key = self.readout_edge_keys[idx]
            # print(f"readout key {key}")
            # print(f"READOUT LAYER {idx+1} out of {len(self.readouts)} for {key}")
            d = data[key]
            src, dst = d.edge_index 
            edge_attr = d.edge_attr
            sh = d.sh
            xsrc = data[key[0]].ft[src]
            readout = conv(dst,xsrc,None,sh,edge_attr, num_graphs=data.num_graphs)
            node_readouts.append(readout)
            # print(f'xsrc for {key} {xsrc}')
            # print(f'edge_attr for {key} {edge_attr}')
            # print(f'node_readout for {key}: {readout}')

            if self.inspect: 
                # print(f"{key[0]} readout \n: {node_readouts[-1]}")
                self.irrep_schedule.append({'name': f"aggregation layer {key}", 'irreps': [conv.irreps_in_src, conv.irreps_sh, conv.irreps_out,conv.irreps_in_dst]})
                self.layer_outputs.append({'name': f"aggregation layer {key}", 'ft': node_readouts[-1].clone()})

        # print(f"node readouts per batch: {node_readouts}")
        out = self.fuse(node_readouts)
            
        return out 
        
    def encode(self, data): 
        """
        alters node and edge features, not edge indices, sh or positions. 
        """
        for name in self.field: 
            name = self.alias_map[name]
            if isinstance(self.node_encoders[name],EncodeCombined): 
                data[name].ft = self.node_encoders[name](data[name].ft, data[name].is_type1) 
            else: 
                data[name].ft = self.node_encoders[name](data[name].ft) 
            # print(f"node encoder {name}, ft now {data[name].ft.shape}")

            # if self.inspect:
            #     print(f"node attribute embedding {name} {data[name].ft.shape}: {data[name].ft}")

        for i, name in enumerate(self.edge_encoders.keys()):
            if "readout" in name.split('-'):
                src, dst = self.alias_to_name(name.split('-'))
                dst = dst[name.split('-')[0]]
                data[src, 'global',dst].edge_attr = self.edge_encoders[name](data[src,'global',dst].edge_attr)
            else: # this is an interaction layer 
                # print(name.split('-')[0]) # name is xx-prox, so name.split('-')[0] is xx, so 
                src = self.alias_map[list(name.split('-')[0])[0]]
                dst = self.alias_map[list(name.split('-')[0])[1]]
                edge_name = name.split('-')[1]
                data[src,edge_name,dst].edge_attr = self.edge_encoders[name](data[src,edge_name,dst].edge_attr)
            
            # if self.inspect:
            #     print(f"edge attribute embedding {name}: {data[src,'conv',dst].edge_attr}")

    def get_node_encoders(self):
        node_encoders = nn.ModuleDict()
        for i in self.field_shorthand: 
            id = self.alias_to_name([i])[0]
            if i in ['p','l','c']: 
                node_encoders[id] = (EncodeAtom(
                    num_atom_type=self.atom_feature_dims[0],
                    hyb_types_num=self.hyb_types_num, hyb_embed_dim=self.hyb_embed_dim,
                    binary_features_num=self.atom_feature_dims[1], binary_embed_dim=self.binary_embed_dim,
                    scalar_features_num=self.atom_feature_dims[3], out_dim = self.ns
            ))  
            elif i in ['r']: 
                node_encoders[id]=(EncodeResidueNodes(self.res_node_dim, self.ns))
            elif i in ['x','z']: 
                node_encoders[id]=(EncodeCombined(num_atom_type=self.atom_feature_dims[0],
                    hyb_types_num=self.hyb_types_num, hyb_embed_dim=self.hyb_embed_dim,
                    binary_features_num=self.atom_feature_dims[1], binary_embed_dim=self.binary_embed_dim,
                    scalar_features_num=self.atom_feature_dims[3], atom_dim_list = self.res_node_dim, ns = self.ns))   
    
        
        return node_encoders 

    def get_edge_encoders(self, dout,dropout): 
        """ 
        input dimension is determined by the data
        output dimension is a hyperparameter
        """
        edge_encoders = nn.ModuleDict() 
        for i in self.edge_field: 
            for j in self.edge_sets[i]: 
                din = self.bond_edge_dim if j in ['lig_bond'] else self.edge_in_dim 
                edge_encoders[f"{i}-{j}"]=(nn.Sequential(
                    nn.Linear(din, dout),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dout, dout)))
            for i in self.field_shorthand: 
                din = self.edge_in_dim 
                # print(f"readout name is {i}-readout")
                edge_encoders[f"{i}-readout"]=(nn.Sequential(
                    nn.Linear(din, dout),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dout, dout))) 
        return edge_encoders

    
    def get_convs(self): 
        """
        Given a schedule (e.g. a single layer 'rr,ll, rl, lr'), define interactiong operations (convolutions)
        and a set of keys to access the heterodata object in forward(). 
        
        For each object in field, borrow virtual node spherical harmonics (computed in preprocessing) to 
        define reaodout convolutions. In forward, "src" in readout_edge_keys is used but not "dst". "dst"
        will be formatted, for example, data['ligand'].readout 
        """ 
        self.convs = nn.ModuleList()
        self.readouts = nn.ModuleList()
        self.edge_keys = [] 
        self.readout_edge_keys = [] 

        l_progress_dict = {} 
        for i in self.field:
            l_progress_dict[self.alias_map[i]] = 0
        for idx,edge_type_str in enumerate(self.schedule):
            for edge_name in self.edge_sets[edge_type_str] : 
                src = self.alias_map[edge_type_str[0]]
                dst = self.alias_map[edge_type_str[1]]

                edge_key = (src, edge_name, dst)
                
                conv_count_dst = l_progress_dict[src]
                conv_count_src = l_progress_dict[dst]
                l_progress_dict[dst] += 1 # progress the receiving l not the sending l. 
                
                l_idx_in_src = conv_count_src if conv_count_src<len(self.l_schedule)-1 else -1
                l_idx_in_dst = conv_count_dst if conv_count_dst<len(self.l_schedule)-1 else -1 
                l_idx_out = conv_count_dst+1 if conv_count_dst<len(self.l_schedule)-1 else -1 
                
                extra_l_src = "+3x1o" if l_idx_in_src == 0 and src in ['protein','protein_ligand','protein_pocket'] else None 
                extra_l_dst = "+3x1o" if l_idx_in_dst == 0 and dst in ['protein','protein_ligand','protein_pocket'] else None 

                l_src = self.l_schedule[l_idx_in_src]+(extra_l_src or "")
                l_dst = self.l_schedule[l_idx_in_dst]+(extra_l_dst or "")
                l_out = self.l_schedule[l_idx_out]

                conv = Conv(l_src, l_dst, self.lsh, self.edgedim, l_out, self.hdim, edge_key=edge_key)
                self.convs.append(conv)
                self.edge_keys.append(edge_key)
            
        # virtual node style graph level readout for all graphs on field
        for i in self.field_shorthand: 
            src = self.alias_map[i]
            dst = self.alias_map['readout'][i]
            # src , dst = self.alias_to_name([f"{i}v"])[0].split('-')
            edge_key = (src,'global',dst)
            # print(f"SOURCE {src}")
            conv_count_src = l_progress_dict[src]
            l_idx_in_src = conv_count_src  if conv_count_src<len(self.l_schedule)-1 else -1 

            # print(f"at readout, the input l is {conv_count_src}")
            readout_l = f"{self.readout_dim}x0e"

            conv = Conv(self.l_schedule[l_idx_in_src], None, self.lsh, self.edgedim, readout_l, self.hdim, edge_key =edge_key)
            self.readouts.append(conv)
            self.readout_edge_keys.append(edge_key)

                
    def get_fuse(self, in_dim, out_dim=None, type ='MLP', num_objects=4):

        if type == 'MLP':
            return MLP(in_dim, out_dim or in_dim, num_objects=num_objects)
        elif type == 'cross_att':
            return CrossAttention(in_dim)
        elif type == 'transformer':
            return Transformer(in_dim)
        else:
            raise ValueError(f"Unknown fusion type: {type}")

    
    def expand_schedule(self): 
        self.schedule = [i for j in self.schedule for i in j]
        # self.field = self.alias_to_name(self.field)
    
    def alias_to_name(self, aliases):
        return [self.alias_map[alias.strip()] for alias in aliases if alias.strip() in self.alias_map]

   
    
    



