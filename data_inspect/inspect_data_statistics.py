from torch_geometric.loader import DataLoader
import pickle
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

def find_interacting_atoms(decoy, target, cutoff=6.0):
    distances = distance.cdist(decoy, target)
    decoy_atoms, target_atoms = np.nonzero(distances < cutoff)
    decoy_atoms, target_atoms = decoy_atoms.tolist(), target_atoms.tolist()
    return decoy_atoms, target_atoms

split = "test"
file = f"../processed_data/refined2016/basic/{split}.pkl"
# file = "../data/cache/val_final.pkl"
with open(file, "rb") as f: 
    test_dataset = pickle.load(f)

batch_size = 1

loader = DataLoader(test_dataset, batch_size = 1, shuffle = True)

cutoffs = [1,2,4,6,8,10,12]
d = None
count = 0
natoms = np.zeros(len(loader))
nca = np.zeros(len(loader))
npk = np.zeros(len(loader))
atoms_per_residue = np.zeros(len(loader))
nlatoms = np.zeros(len(loader))
pkl_dist = np.zeros(len(loader))
pk_interact = np.zeros((len(cutoffs),len(loader)))
l_pk_interact = np.zeros((len(cutoffs),len(loader)))
pna_interact = np.zeros((len(cutoffs),len(loader)))
l_pna_interact= np.zeros((len(cutoffs),len(loader)))
pn_interact = np.zeros((len(cutoffs),len(loader)))
l_pn_interact= np.zeros((len(cutoffs),len(loader)))


for idx, d in enumerate(tqdm(loader)):
    pn = d["protein"].cpu()
    pk = d["pocket"].cpu()
    pna = d["protein_all"].cpu()
    ln = d["ligand"].cpu()

    no_all_atom = True if len(pna)==0 else False
    
    if len(pk)==0:
        print("no pk")
    if len(ln)==0:
        print("no ln")
    if len(pn)==0:
        print("no pn")

    # number of atoms 
    nca[idx] = pn['pos'].shape[0]
    natoms[idx] = pna['pos'].shape[0] if not no_all_atom else 0
    atoms_per_residue[idx] = natoms[idx]/nca[idx]
    npk[idx] = pk['pos'].shape[0]
    nlatoms[idx] = ln['pos'].shape[0]

    # mean position 
    pn_pos_mean = torch.mean(pn['pos'],axis=0).cpu()
    pk_pos_mean = torch.mean(pk['pos'],axis=0).cpu()
    ln_pos_mean = torch.mean(ln['pos'],axis=0).cpu()
    pkl_dist[idx] = torch.norm(pk_pos_mean - ln_pos_mean)
    
    
    for cutoff_id,cutoff in enumerate(cutoffs): 
        # interacting atoms in the pocket
        pk_inter, l_inter = find_interacting_atoms(pk['pos'],ln['pos'], cutoff)
        pk_interact[cutoff_id,idx] = len(set(pk_inter))
        l_pk_interact[cutoff_id,idx] = len(set(l_inter))

        # interacting atoms from all-atom 
        pna_inter, l_inter= find_interacting_atoms(pna['pos'],ln['pos'], cutoff) if not no_all_atom else [0,0]
        pna_interact[cutoff_id,idx] = len(set(pna_inter)) if not no_all_atom else 0
        l_pna_interact[cutoff_id,idx] = len(set(l_inter)) if not no_all_atom else 0
   
        # interacting atoms from residue 
        pn_inter, l_inter = find_interacting_atoms(pn['pos'],ln['pos'], cutoff)
        pn_interact[cutoff_id,idx] = len(set(pn_inter))
        l_pn_interact[cutoff_id,idx] = len(set(l_inter))


mean_natoms = np.mean(natoms)
var_natoms = np.std(natoms) 
mean_nca = np.mean(nca)
var_nca = np.std(nca)

for cutoff_id,cutoff in enumerate(cutoffs): 
    print(f'mean number of pocket atoms within {cutoff}A of ligand is {np.mean(pk_interact[cutoff_id])}')
    print(f'mean number of ligand atoms within {cutoff}A of pocket is {np.mean(l_pk_interact[cutoff_id])}')
    print(f'mean number of protein atoms within {cutoff}A of ligand is {np.mean(pna_interact[cutoff_id])}')
    print(f'mean number of ligand atoms within {cutoff}A of protein is {np.mean(l_pna_interact[cutoff_id])}')
    print(f'mean number of residue atoms within {cutoff}A of ligand is {np.mean(pn_interact[cutoff_id])}')
    print(f'mean number of ligand atoms within {cutoff}A of residue is {np.mean(l_pn_interact[cutoff_id])}')

print('\naverage mean distance between ligand and pocket', np.mean(pkl_dist))
print('with std', np.std(pkl_dist))
print('\nmean number of atoms ', mean_natoms)
print('with std ', var_natoms)
print('\nmean number of residues ', mean_nca)
print('with std ', var_nca)
print('\nmean atoms per residue', np.mean(atoms_per_residue))
print('with std', np.std(atoms_per_residue))
print('\nmean number of pocket atoms ', np.mean(npk))
print('with std', np.std(npk))



# pn = d["protein"]
# print(pn.ft.shape)
# print(pn.ft[0])



# pk = d["pocket"].pos
# pos = d["protein"].pos
# ft = d["protein"].ft

# print(ft)
# s1 = ft[:,:6] # dihedral scalars 
# x1 = ft[:,6:9] # ca orientation 1 
# x2 = ft[:,9:12] # ca orientation 2 
# x3 = ft[:,12:15] # sidechain orientation 


# import matplotlib.pyplot as plt
# plt.plot(x1.cpu().numpy())


# le = d["ligand","ligand"]
# print(le.edge_attr.shape) # should be Nedges, 12

# ln = d["ligand"]
# print(ln.ft.shape)
# print(ln.ft)





# ps = d["pocket_surface"]
# print(ps)
