

""" 
Given a set of keys from net_logic.py corresponding to parts of the molecular complex 
1. Place separate graphs onto a shared coodinate system .
2. Merge edge indices into one 
3. Coarsen graph for a set of nonglobal virtual nodes if needed 
4. Spawn virtual nodes + corresponding edge indices 


1. edge indices are sufficient, they can be used to train three se
2. virtual nodes can be placed by the network as long as a we have a set of edge indices. 
3. Since readouts need a global node, we'll build global edge indices no matter what. 
4. However, if virtual nodes corresponding to subgraphs are needed, we need to build those here. 
5. If we build subgraphs, we also need to build global nodes for those subgraphs.  

We can place everything next to each other and simply use N different networks. 



Design 
1. Individual distinct functions have been written to build each type of graph. 
2. Complexes only include interacting atoms determined by the radius. 


Virtual node
Obtain a set of edge indices mapping each structure to a virtual node. 
Come training, simply define a variable corresponding to virtual nodes. 

Some conventions 
in pl_node, protein first ligand next



"""

import torch 
from net_logic import field 
from models_mp.build_graph import (build_ligand_graph, build_pl_graph, build_protein_graph, build_l_readout, build_p_readout, build_pl_readout)
from e3nn import o3 
from functools import partial
from build_graph_ import build_graph, build_readout 
from torch_scatter import scatter_mean 


OBJECT_MAP = {
    'l': ['ligand'],
    'r': ['protein'],  # residue
    'p': ['pocket'],
    'x': ['protein', 'ligand'],
    'c': ['ligand', 'pocket'],
    'z': ['pocket', 'protein']
}

EDGE_ATTR = {
    'l': lambda d, ctx: ligand_edge_attr(d, ctx['max_radius'], ctx['edge_dim'])
}


class Transform(torch.nn.Module): 
    def __init__(self, args, shl=2, smoothed_dim = 32): 
        super().__init__() 
        self.ids = field 
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=shl)
        self.device = "cuda:0"
        
        self.edge_types = {'all': ['prox','lig_bond'],
            'r': ['prox'],
            'l': ['prox','lig_bond'],
            'p': ['prox'],
            'x': ['prox'],
            'z': ['prox'],
            'c': ['prox'],
            }


        self.contact_distances = {
            'l': args.ld,
            'r': args.rd,  
            'p': args.pd,
            'x': args.rld,
            'c': args.pld,
            'z': args.rpd
        }

        self.center_max_distances = {
            'l': args.lcd,
            'r': args.rcd,  
            'p': args.pcd,
            'x': args.rlcd,
            'c': args.plcd,
            'z': args.rpcd
        }

        self.max_neighbors = {
            'l': args.l_max_nb,
            'p': args.p_max_nb,
            'r': args.r_max_nb,
        }

        self.distance_smoothing = {} 
        self.center_distance_smoothing ={} 
        for id in field: 
            self.distance_smoothing[id] = GaussianSmearing(0.0,self.contact_distances[id], smoothed_dim).to(self.device)
            self.center_distance_smoothing[id] = GaussianSmearing(0.0,self.center_max_distances[id], smoothed_dim).to(self.device)

    def build_multiplex(self,d): 
        for key in self.ids: 
            self.add_edges(d,key) 
        self.clean_up_edges(d,key)
        print(d)
        return d 
    

    def set_edge_attributes(self, d,ei,attr,sh):
        """
        d should be in the form of g[a,'conv',b]
        """
        d.edge_index = ei
        d.sh = sh 
        d.edge_attr = attr 

    def fuse_graphs(self,d,name,name1,name2,ei,ea,sh, edge_name):

        """
        get contact indices, create combined object
        """

        def pad_features(ft, target_dim):
            pad_dim = target_dim - ft.size(1)
            if pad_dim > 0:
                x = torch.cat([ft, torch.zeros(ft.size(0), pad_dim, device=ft.device)], dim=1)
                return x
            return ft
        
        i2,i1, inter_ei = self.get_contacts(ei)

        #nodes
        if name not in d.node_types:
            d1 = d[name1]
            d2 = d[name2]
            n1 = d1.pos[i1].size(0)
            n2 = d2.pos[i2].size(0)
            d[name].pos = torch.cat([d1.pos[i1], d2.pos[i2]], dim=0)
            ftd = max(d1.ft.shape[1],d2.ft.shape[1])
            d[name].ft = torch.cat([pad_features(d1.ft[i1],ftd),pad_features(d2.ft[i2],ftd)],dim=0)
            d[name].is_type1  = torch.cat([
                        torch.ones(n1, dtype=torch.bool),
                        torch.zeros(n2, dtype=torch.bool)
                    ])

       
        #edges
        d1e = d[name1,edge_name,name1]
        d2e = d[name2,edge_name,name2]
        sub_ei1, sub_ea1, sub_sh1 = extract_subgraph_edges(d1e.edge_index,d1e.edge_attr, d1e.sh, i1) 
        sub_ei2, sub_ea2, sub_sh2 = extract_subgraph_edges(d2e.edge_index,d2e.edge_attr, d2e.sh, i2) 
        d[name,edge_name,name].edge_index = torch.cat([sub_ei1, sub_ei2 + n1, inter_ei], dim=1)
        d[name,edge_name,name].edge_attr = torch.cat([sub_ea1, sub_ea2,ea], dim=0)
        d[name,edge_name,name].sh = torch.cat([sub_sh1,sub_sh2,sh], dim=0)

        
    def get_contacts(self,ei):         
        a = torch.unique(ei[0])
        b = torch.unique(ei[1])

        na = a.size(0)
        nb = b.size(0)

        # Map: original indices → new indices
        a_map = -torch.ones(ei[0].max().item() + 1, dtype=torch.long, device=ei.device)
        b_map = -torch.ones(ei[1].max().item() + 1, dtype=torch.long, device=ei.device)

        a_map[a] = torch.arange(na, device=ei.device)
        b_map[b] = torch.arange(nb, device=ei.device) + na  # shifted

        i = a_map[ei[0]]
        j = b_map[ei[1]]
        inter_ei = torch.stack([i, j], dim=0)
        
        return a, b, inter_ei
    

    def add_edges(self,g,key):
        a,b,c,d= self.key_to_name(key) # l,p,r,z,x to name 
        for edge_name in self.edge_types[key]: 
            ei,attr,sh = self.get_edges(key,g,edge_name)
            # combine objects as needed 
            if d is not None:
                self.fuse_graphs(g,name=d,name1=a,name2=b,ei=ei,ea=attr,sh=sh, edge_name=edge_name)
            # or first build graphs for distinct objects
            if d==None: 
                self.set_edge_attributes(g[a,edge_name,a], ei,attr,sh)

        # add center-of-mass node to each object 
        self.get_center_of_mass_graph(key,g)

           
    def key_to_name(self,key):
        mapping = {
                    'r': ['protein','protein','virtual_r',None],
                    'x': ['protein','ligand','virtual_x', 'protein_ligand'],
                    'z': ['protein', 'pocket','virtual_z','protein_pocket'], 
                    'l': ['ligand','ligand','virtual_l',None],
                    'p': ['pocket','pocket', 'virtual_p',None],
                    'c': ['pocket','ligand','virtual_c','pocket_ligand']
                }
        a = mapping[key][0]
        b = mapping[key][1]
        c = mapping[key][2]
        d = mapping[key][3]
        return a,b,c,d 
        
    def get_edges(self, key, d, edge_name): 
        """ 
        build graph first, then add edge attributes 
        """ 
        compute_sh = True 
        compute_radial = True
        compute_length_emb = True

        max_radius = self.contact_distances[key]
        try:
            max_neighbors = self.max_neighbors[key]
        except: 
            max_neighbors = 10000  
        smooth = self.distance_smoothing[key]
        name_src = self.key_to_name(key)[0]
        name_dst = self.key_to_name(key)[1] 

        src_pos = d[name_src].pos
        dst_pos = d[name_dst].pos 
        # if distinct object, find ones with bonds. Set to none if no bonds or is combined. 
    
        if name_src == name_dst: 
            try: 
                edge_index = d[name_src,edge_name,name_src].edge_index
                edge_attr = d[name_src,edge_name,name_src].edge_attr 
                compute_sh = True 
                compute_radial = False 
                compute_length_emb = False 
                # need to add an option to not compute radius graph and just compute sh from existing edges. 
                # or not compute sh at all. That's the most general. 
            # print(f'pre-build edge index {edge_index.shape} and edge_attr {edge_attr.shape}')
            except:
                edge_index = edge_attr = None
        else: 
            edge_index = edge_attr = None 


        graph_attributes = build_graph(src_pos, dst_pos, edge_index = edge_index, 
                                # batch_src=batch_src, batch_dst=batch_dst,
                                max_radius=max_radius, max_neighbors=max_neighbors,
                                edge_attr=edge_attr, distance_fn=smooth, sh_irreps=self.sh_irreps, compute_sh = compute_sh,
                                compute_radial = compute_radial, compute_length_emb = compute_length_emb)  


        return graph_attributes

    def get_center_of_mass_graph(self,key,g):
        a,b,c,combined_name = self.key_to_name(key)
        name = combined_name if combined_name is not None else a 
        center_smooth = self.center_distance_smoothing[key]
        g_ = g[name]
        # center = scatter_mean(g_.pos, dim=0)
        center = torch.mean(g_.pos, dim=0).unsqueeze(1)
        edge_index, attr, sh = build_readout(g_.pos, center, self.sh_irreps,center_smooth)
        g[c].pos = center.T # 3,1 -> 1,3 so that it's not treated as three nodes 
        # print(f"edge_index for {key} {name}: max {edge_index.max()}")
        self.set_edge_attributes(g[name,'global',c], edge_index.flip(0), attr, sh)


        
    def clean_up_edges(self,d,key):
        if 'ft_names' in d[self.key_to_name(key)[0]]:
                    del d[self.key_to_name(key)[0]]['ft_names']

        for edge_type in d.edge_types:
            if edge_type[1] not in self.edge_types['all'] and edge_type[1]!='global':
                del d[edge_type]

def extract_subgraph_edges(ei, ea, sh, sub_nodes):
    # Create a mask for which nodes are in the subgraph
    node_mask = torch.zeros(ei.max()+1, dtype=torch.bool, device=ei.device)
    sub_nodes = sub_nodes[sub_nodes<=ei.max()]
    node_mask[sub_nodes] = True

    # Select edges where both src and dst are in sub_nodes
    src, dst = ei
    # print(f"node mask {node_mask}, src {src}, dst {dst}")
    edge_mask = node_mask[src] & node_mask[dst]
    # Filter edges and attributes
    sub_ei = ei[:, edge_mask]
    sub_ea = ea[edge_mask]
    sub_sh = sh[edge_mask] if sh is not None else None

    # Remap global node indices to local ones in the subgraph
    idx_map = torch.full((ei.max()+1,), -1, dtype=torch.long, device=ei.device)
    idx_map[sub_nodes] = torch.arange(len(sub_nodes), device=ei.device)
    sub_ei = idx_map[sub_ei]

    return sub_ei, sub_ea, sub_sh


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()

        # Define learnable radius, using log to ensure positivity
        # log_stop = torch.nn.Parameter(torch.log(torch.tensor(stop)))
        # stop = torch.exp(log_stop)

        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        # print("dist device", dist.device)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))



class CachedTransformDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform_fn, device=None, safety_checks = False ):
        self.base = base_dataset
        self.transform = transform_fn
        self.cache = [None] * len(base_dataset)
        self.device = device
        self.safety_checks = safety_checks 


    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):

        # clone to avoid destructive transforms
        if self.cache[idx] is None:
            data = self.base[idx].to("cuda:0") #.clone()
            # print("data device", data['pocket'].ft.device)
            # data = data.to(self.device) if self.device else data  
            with torch.no_grad():
                data = self.transform(self.base[idx])
            self.cache[idx] = data


        if self.safety_checks: 
            print(f"\n>> IDX {idx}")
            for edge_type in data.edge_types:
                edge_store = data[edge_type]
                for key, val in edge_store.items():
                    print(f"edge[{edge_type}][{key}] = {None if val is None else val.shape}")

            for node_type in data.node_types:
                node_store = data[node_type]
                for key, val in node_store.items():
                    try:
                        print(f"node[{node_type}][{key}] = {None if val is None else val.shape}")
                    except: 
                        print(f"node[{node_type}][{key}]={len(val) if val is not None else None }")


                        # Defensive checks
            for node_type in data.node_types:
                store = data[node_type]
                for k, v in store.items():
                    if isinstance(v, torch.Tensor) and v.numel() == 0:
                        print(f"Empty tensor: node[{node_type}][{k}]")

            for edge_type in data.edge_types:
                store = data[edge_type]
                for k, v in store.items():
                    if isinstance(v, torch.Tensor) and v.numel() == 0:
                        print(f"Empty tensor: edge[{edge_type}][{k}]")

            for edge_type in data.edge_types:
                for k, v in data[edge_type].items():
                    try:
                        _ = v.size()
                    except Exception as e:
                        print(f"edge[{edge_type}][{k}] failed .size(): {e}")
                    else:
                        print(f"edge[{edge_type}][{k}] shape: {v.shape}")

            for node_type in data.node_types:
                for k, v in data[node_type].items():
                    try:
                        _ = v.size()
                    except Exception as e:
                        print(f"node[{node_type}][{k}] failed .size(): {e}")
                    else:
                        print(f"node[{node_type}][{k}] shape: {v.shape}")
                    
        return self.cache[idx].to(self.device) if self.device else self.cache[idx]


