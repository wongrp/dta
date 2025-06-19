import torch
from e3nn.o3 import rand_matrix
from e3nn.o3 import Irreps

from copy import deepcopy
from net import Net
import pickle
from torch_geometric.data import DataLoader
from net_logic import field,schedule,l_schedule, args
from net_transform_graphs import CachedTransformDataset, Transform



class EquivarianceTest:
    def __init__(self, model, device="cuda", vec_field="ft", pos_key="pos", vec_starts=(6, 9, 12)):
        self.model = model
        self.device = device
        self.vec_field = vec_field
        self.pos_key = pos_key
        self.vec_starts = vec_starts 

    def rotate_data(self, data, R):
        data_rot = deepcopy(data)

        # Rotate all position vectors (for all node types)
        for node_type in data.node_types:
            print(f'rotate node type: {node_type}')
            data_rot[node_type].pos = data[node_type].pos @ R.T 
            print(f"{data_rot[node_type].pos[0]} from {data[node_type].pos[0]}")

        # Rotate orientation vectors only for 'protein'
        if "protein" in data.node_types and self.vec_field in data["protein"]:
            for start in self.vec_starts:
                print(f"start {start}")
                data_rot["protein"].ft[:, start:start+3] = (
                    data["protein"].ft[:, start:start+3] @ R.T 
                )
        
        D = self.model.lsh.D_from_matrix(R.cpu()).to(self.device)
        for edge_type in data.edge_types:
            if 'conv' in edge_type: 
                print(f"edge type {edge_type}")
                data_rot[edge_type].sh = data[edge_type].sh @ D.T 

            

        return data_rot

    def get_irreps(self):
        irreps = [Irreps(x.irreps_out) for x in self.model.convs + self.model.readouts]
        return irreps
 
    def test(self, data):
        data = data.to(self.device)
        self.model.eval()
        self.irreps = self.get_irreps()

        with torch.no_grad():
            R = rand_matrix(device = self.device)
            data_rot = self.rotate_data(data, R)
            _ = self.model(data)
            clean_acts = deepcopy(self.model.layer_outputs) # clone, otherwise they will be same
            print("running model on rotated data")
            _ = self.model(data_rot)
            rot_acts = deepcopy(self.model.layer_outputs)
            irrep_schedule = self.model.irrep_schedule

        diffs = {}
        for i, (clean, rot) in enumerate(zip(clean_acts, rot_acts)):
            layer_diff = {}
            node_type = clean['name']
            if clean['ft'].shape != rot['ft'].shape:
                print("shape mismatch")
                continue    
            print(f"node type {node_type}")
            print(f"{node_type} layer {i} output clean: {clean['ft'][0,:5]}")
            print(f"{node_type} layer {i} output rot: {rot['ft'][0,:5]}")

            D = self.irreps[i].D_from_matrix(R.cpu()).to(self.device)
            rotated_clean = (clean['ft']@ D.T)
            print(rotated_clean.shape)
            print(rot['ft'].shape)
            diff = torch.norm(rotated_clean - rot['ft'])

            layer_diff[node_type] = diff
            diffs[f"layer_{i}"] = layer_diff

        return diffs, irrep_schedule 


class InstrumentedNet(Net):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_outputs = []

    def forward(self, data):
        self.encode(data)
        self.layer_outputs.clear()

        for idx, conv in enumerate(self.convs):
            key = self.edge_keys[idx]
            print(f"CONV LAYER {idx+1} out of {len(self.convs)} for {key}")
            d = data[key]
            src, dst = d.edge_index
            edge_attr = d.edge_attr
            sh = d.sh

            xsrc = data[key[0]].ft[src]
            xdst = data[key[2]].ft
            data[key[2]].ft = conv(dst, xsrc, xdst, sh, edge_attr)

            # Store only the updated node type after this conv
            self.layer_outputs.append({key[2]: data[key[2]].ft.clone()})

        print("output keys",[x.keys() for x in self.layer_outputs])

        node_readouts = []
        for idx, conv in enumerate(self.readouts):
            key = self.readout_edge_keys[idx]
            print(f"READOUT LAYER {idx+1} out of {len(self.readouts)} for {key}")
            d = data[key]
            src, dst = d.edge_index
            edge_attr = d.edge_attr
            sh = d.sh
            xsrc = data[key[0]].ft[src]
            node_readouts.append(conv(dst, xsrc, None, sh, edge_attr, num_graphs=data.num_graphs))
            self.layer_outputs.append({key: node_readouts[-1].clone()})

        out = self.fuse(node_readouts)
        return out



test_set_path = "processed_data/refined2016/basic/test.pkl"
device = "cuda:7" if torch.cuda.is_available() else "cpu"


with open(test_set_path, "rb") as f:
    ds = pickle.load(f)
transform = Transform(args)
ds = CachedTransformDataset(ds, transform.build_multiplex, device)
loader = DataLoader(ds, shuffle=True)

print(ds[0])
net = Net(args, field, schedule, l_schedule).to(device)
net.inspect = True 


sample = next(iter(loader)).to(device)
tester = EquivarianceTest(net, device=device)
layer_diffs, irrep_schedule = tester.test(sample)


for layer, diffs in layer_diffs.items():
    print(f"{layer}:")
    for node_type, diff in diffs.items():
        print(f"  {node_type}: {diff}")# {diff:.2e}")

for idx, layer_irreps in enumerate(irrep_schedule):
    print(layer_irreps)
    ir = layer_irreps['irreps']
    name = layer_irreps['name']
    print(f"layer {idx} {name} irreps: (({ir[0]}) ox ({ir[1]})) -> {ir[2]})+{ir[3]}")