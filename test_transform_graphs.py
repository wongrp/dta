
import pickle 
from torch_geometric.loader import DataLoader
from args import get_args
from net_transform_graphs import Transform


with open("data/cache/test_final.pkl", "rb") as f: 
    ds = pickle.load(f)

device = 'cuda:0'
args = get_args()
loader = DataLoader(ds, batch_size = 2, shuffle=True)


transform = Transform(args)
for d in loader: 
    d.to(device)
    transform.build_multiplex(d)

    # print(d['protein'])
    # print(d['virtual_r'])
    # print(d['pocket'])



    #    'r': ['protein','protein','virtual_r'],
    #                 'rl': ['protein','ligand','virtual_rl'],
    #                 'l': ['ligand','ligand','virtual_l'],
    #                 'p': ['pocket','pocket', 'virtual_pk'],
    #                 'c': ['pocket','ligand','virtual_c'],

          
    break