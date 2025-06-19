
import torch
from e3nn import o3 
from torch_cluster import radius, radius_graph



def build_graph(
    src_pos, dst_pos, edge_index = None, 
    batch_src=None, batch_dst=None,
    max_radius=None, max_neighbors=None,
    edge_attr=None, distance_fn=None, sh_irreps=None, compute_length_emb =True, 
    compute_radial = True, compute_sh = True 
):
    if compute_radial: 
        # compute edge index, edge if inside radius, not edge if out. 
        radius_edge_index = radius(src_pos, dst_pos, max_radius, batch_src, batch_dst, max_num_neighbors=max_neighbors)
        # if edge indices already exist, such as bonds, concatenate. 
        edge_index = torch.cat([edge_index, radius_edge_index], 1).long() if edge_index is not None else radius_edge_index

        
   
    receiver,sender  = edge_index
    # compute smoothed displacement vectors 
    edge_vec = dst_pos[receiver] - src_pos[sender]
    if compute_length_emb:  
        edge_len_emb = distance_fn(edge_vec.norm(dim=-1))
        # if edge attributes already exist, such as bond type, concatenate. 
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_len_emb], dim=1)
        else:
            edge_attr = edge_len_emb
        
    edge_sh = None 
    if compute_sh: 
        edge_sh = o3.spherical_harmonics(sh_irreps, edge_vec, normalize=True, normalization="component")

    
    return edge_index, edge_attr, edge_sh


def build_readout(
    node_pos, center_pos, sh_irreps, distance_fn, batch=None
):
    batch = torch.zeros(node_pos.shape[0], dtype=torch.long, device = node_pos.device) if batch is None else batch 
    edge_index = torch.stack([batch, torch.arange(len(batch), device=node_pos.device)])
    edge_vec = node_pos - center_pos[batch]
  
    edge_attr = distance_fn(edge_vec.norm(dim=-1))
    edge_sh = o3.spherical_harmonics(sh_irreps, edge_vec, normalize=True, normalization="component")
    return edge_index, edge_attr, edge_sh


def build_readout_edge_index(d):
    """
    Constructs edge_index mapping graph-wise centers to each node in d.
    Assumes:
    - If d.batch exists, it's batched data.
    - If not, it's a single graph.
    """
    device = d.pos.device
    num_nodes = d.pos.size(0)

    if hasattr(d, 'batch'):
        batch = d.batch
    else:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)  # all nodes in graph 0

    node_indices = torch.arange(num_nodes, device=device)
    return torch.stack([batch, node_indices], dim=0)






def get_build_graph(self, key):
        objs = OBJECT_MAP[key]
        def get_pos(d): return [d[o].pos for o in objs]
        def get_batch(d): return [d[o].batch for o in objs]

        def compute_center(pos_list, batch_list):
            pos_cat = torch.cat(pos_list, dim=0)
            batch_cat = torch.cat(batch_list, dim=0)
            return torch_scatter.scatter_mean(pos_cat, batch_cat, dim=0)

        def graph_fn(d):
            pos = get_pos(d)
            batch = get_batch(d)
            pos_cat = torch.cat(pos, dim=0)
            batch_cat = torch.cat(batch, dim=0)
            ctx = {
                'max_radius': getattr(self, f"{key}d"),
                'edge_dim': 12  # change if needed
            }
            edge_attr_fn = EDGE_ATTR.get(key, lambda d, ctx: None)
            return build_graph(
                src_pos=pos_cat,
                dst_pos=pos_cat,
                batch_src=batch_cat,
                batch_dst=batch_cat,
                max_radius=ctx['max_radius'],
                max_neighbors=10000,
                edge_attr=edge_attr_fn(d, ctx),
                distance_fn=getattr(self, f"smooth_{key}"),
                sh_irreps=self.sh_irreps
            )

        def readout_fn(d):
            pos = get_pos(d)
            batch = get_batch(d)
            pos_cat = torch.cat(pos, dim=0)
            batch_cat = torch.cat(batch, dim=0)
            center = torch_scatter.scatter_mean(pos_cat, batch_cat, dim=0)
            return build_readout(pos_cat, batch_cat, center, self.sh_irreps, getattr(self, f"smooth_{key}c")), center

        return graph_fn, readout_fn