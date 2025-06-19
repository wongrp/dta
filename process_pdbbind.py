import os
import torch
import argparse 
import openbabel
from openbabel import pybel
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.data import Data,HeteroData
from scipy.spatial import distance_matrix
import torch_geometric.transforms as T
import pickle
from tqdm import tqdm
from transformers import pipeline
import re
from prody import *
import networkx as nx
import numpy as np
from transformers import BertModel, BertTokenizer
import os
from Bio.PDB import *
import atom3d.util.formats as fo 
from utils.protein_utils import featurize_as_graph
from utils.openbabel_featurizer import Featurizer
from process_pdbbind_utils import GetPDBDict, GetPDBList, GetPDBList_core, GetPDBDict_core

"""
guide 
Q: Where do pocket atom feats come from? 
A: read_protein()->featurizer.get_features(protein_pocket)->atom_fea 
"""
# for one hot 
ele2num = {'H':0, 'LI':1, 'C':2, 'N':3, 'O':4, 'NA':5, 'MG':6, 'P':7, 'S':8, 'K':9, 'CA':10, 'MN':11, 
           'FE':12, 'CO':13, 'NI':14, 'CU':15, 'ZN':16, 'SE':17, 'SR':18, 'CD':19, 'CS':20, 'HG':21, 'RB':22}


def info_3D(a, b, c):
    ab = b - a
    ac = c - a
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


def load_all_atom_structure(fname):
    # Load the data
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    residue_to_index = {}
    atom_residue_indices = []
    coords = []
    types = []
    res_index = 0 # starts at 0 

    # Loop through residues and assign an index
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip if this is a hetero atom (like water or ligand) and you want only standard residues
                # from Bio.PDB.Polypeptide import is_aa
                # if not is_aa(residue): continue

                residue_to_index[residue] = res_index
                for atom in residue:
                    atom_residue_indices.append(res_index)
                    coords.append(atom.get_coord())
                    types.append(ele2num[atom.element])
                res_index += 1


    # for atom in atoms:
    #     coords.append(atom.get_coord())
    #     types.append(ele2num[atom.element])
       
    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    return coords, types_array, atom_residue_indices
    
def load_protein_atoms(protein_path, item):
    # try:
    atom_coords, atom_types, atom_residue_indices = load_all_atom_structure(protein_path+item+"_protein.pdb")
    # except:
    #     return None
    # atom_coords, atom_types = load_all_atom_structure(protein_path+item+"_protein.pdb")
    # protein_data = Data(
    #     atom_coords=torch.tensor(atom_coords),
    #     atom_types=torch.tensor(atom_types,dtype=torch.float32),
    # )
    return atom_coords, atom_types, atom_residue_indices


def load_protein_graph(protein_path, item):

    # path = protein_path+item+"_pocket.pdb"
    path = protein_path+item+"_protein.pdb"
    try: 
        protein_df = fo.bp_to_df(fo.read_pdb(path))
        X_ca, seq, node_s, node_v, edge_s, edge_v, edge_index  = featurize_as_graph(protein_df)
    except Exception as e:
        print(f"Error featurizing protein as graph {path}: {e}")
        return None

    return X_ca, seq, node_s, node_v, edge_s, edge_v, edge_index 



def read_ligand(filepath):
    featurizer = Featurizer(save_molecule_codes=False)
    ligand = next(pybel.readfile("mol2", filepath))
    ligand_coord, atom_fea,h_num, atom_ft_names = featurizer.get_features(ligand)
    ligand_center = torch.tensor(ligand_coord).mean(dim=-2, keepdim=True)

    return ligand_coord, atom_fea, ligand, h_num, ligand_center, atom_ft_names

def read_protein(filepath,prot_lm=None):
    featurizer = Featurizer(save_molecule_codes=False)
    protein_pocket = next(pybel.readfile("pdb", filepath))
    pocket_coord, atom_fea,h_num, atom_ft_names = featurizer.get_features(protein_pocket)

    aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'}
    pro_seq_emb = None 
    if prot_lm is not None: 
        seq = ''
        protein_filepath = filepath.replace('_pocket','_protein')
        for line in open(protein_filepath):
            if line[0:6] == "SEQRES":
                columns = line.split()
                for resname in columns[4:]:
                    if resname in aa_codes:
                        seq = seq + aa_codes[resname] + ' '
        sequences_Example = re.sub(r"[UZOB]", "X", seq)

        embedding = prot_lm(sequences_Example)
        e1 = torch.tensor(embedding[0])
        pro_seq_emb = e1

    return pocket_coord, atom_fea,protein_pocket,h_num,pro_seq_emb, atom_ft_names 


def get_complex_edge_fea(edge_list,coord_list):

    net = nx.Graph()
    net.add_weighted_edges_from(edge_list)
    edges_fea = []
    for edge in edge_list:
        edge_fea = []
        edge_fea.append(edge[2])
        edges_fea.append(edge_fea)

    return edges_fea

def compute_pocket_and_ligand_graph(lig_coord, pocket_coord, lig_atom_fea, pocket_atom_fea, score, cut=5):
    coord_list = []
    for atom in lig_coord:
        coord_list.append(atom)
    for atom in pocket_coord:
        coord_list.append(atom)

    dis = distance_matrix(x=coord_list, y=coord_list)
    lenth = len(coord_list)
    edge_list = []

    edge_list_fea = []
    # Bipartite Graph; i belongs to ligand, j belongs to protein
    for i in range(len(lig_coord)):
        for j in range(len(lig_coord), lenth):
            if dis[i, j] < cut:
                edge_list.append([i, j-len(lig_coord), dis[i, j]])
                edge_list_fea.append([i,j,dis[i,j]])

    data = HeteroData()
    edge_index = edgelist_to_tensor(edge_list)

    # ligand 
    data['ligand'].ft = torch.tensor(lig_atom_fea, dtype=torch.float32) # ligand atom features 
    data['ligand'].pos = torch.tensor(lig_coord, dtype = torch.float32)
    data['ligand'].score = torch.tensor(score) # affinity 
    data['pocket'].ft = torch.tensor(pocket_atom_fea, dtype=torch.float32) # pocket features 
    data['pocket'].pos = torch.tensor(pocket_coord, dtype= torch.float32)

    data['ligand', 'pocket'].edge_index = edge_index

    complex_edges_fea = get_complex_edge_fea(edge_list_fea,coord_list)
    edge_attr = torch.tensor(complex_edges_fea, dtype=torch.float32)
    data['ligand', 'pocket'].edge_attr = edge_attr
    data = T.ToUndirected()(data)

    return data


def edgelist_to_tensor(edge_list):
    row = []
    column = []
    coo = []
    for edge in edge_list:
        row.append(edge[0])
        column.append(edge[1])

    coo.append(row)
    coo.append(column)

    coo = torch.Tensor(coo)
    edge_tensor = torch.tensor(coo, dtype=torch.long)
    return edge_tensor


def get_ligand_bonds(lig_atoms_fea,ligand,h_num,score):
    # everything except ligand atom position 
    edges = []
    edges_fea = []
    for bond in openbabel.OBMolBondIter(ligand.OBMol):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 1) or (atom2.GetAtomicNum() == 1):
            continue
        else:
            idx_1 = atom1.GetIdx() - h_num[atom1.GetIdx()-1] - 1
            idx_2 = atom2.GetIdx() - h_num[atom2.GetIdx()-1] - 1

            edge_fea = bond_fea(bond , atom1, atom2)
            edge = [idx_1, idx_2]
            edges.append(edge)
            edges_fea.append(edge_fea)

            re_edge = [idx_2, idx_1]
            edges.append(re_edge)
            edges_fea.append(edge_fea)

    edge_attr = torch.tensor(edges_fea, dtype=torch.float32)
    edge_index = edgelist_to_tensor(edges)

    return edge_attr, edge_index 

def bond_fea(bond,atom1,atom2):
    is_Aromatic = int(bond.IsAromatic())
    is_inring = int(bond.IsInRing())
    d = atom1.GetDistance(atom2)

    node1_idx = atom1.GetIdx()
    node2_idx = atom2.GetIdx()

    neighbour1 = []
    neighbour2 = []
    for neighbour_atom in openbabel.OBAtomAtomIter(atom1):
        if (neighbour_atom.GetAtomicNum() != 1 ) and (neighbour_atom.GetIdx() != node2_idx) :
            neighbour1.append(neighbour_atom)

    for neighbour_atom in openbabel.OBAtomAtomIter(atom2):
        if ( neighbour_atom.GetAtomicNum() != 1) and (neighbour_atom.GetIdx() != node1_idx):
            neighbour2.append(neighbour_atom)

    if len(neighbour1) == 0 and len(neighbour2) == 0:
        return [d,0, 0, 0, 0, 0, 0, 0, 0, 0,is_Aromatic,is_Aromatic]

    angel_list = []
    area_list = []
    distence_list = []

    node1_coord = np.array([atom1.GetX(),atom1.GetY(),atom1.GetZ()])
    node2_coord = np.array([atom2.GetX(),atom2.GetY(),atom2.GetZ()])

    for atom3 in neighbour1:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angel, area, distence = info_3D(node1_coord, node2_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    for atom3 in neighbour2:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angel, area, distence = info_3D(node2_coord, node1_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    return [d,
        np.max(angel_list) * 0.01, np.sum(angel_list) * 0.01, np.mean(angel_list) * 0.01,
        np.max(area_list), np.sum(area_list), np.mean(area_list),
        np.max(distence_list) * 0.1, np.sum(distence_list) * 0.1, np.mean(distence_list) * 0.1,
        is_Aromatic, is_inring]


def process_raw_data(dataset_path,res, processed_file, protein_list, device_id= 0, use_lm = False,all_atom = False, no_from_cache = False, checkpoint_cache_path = 'data/cache/checkpoint_more'):

    G_list = []
    Gaa_list = [] 

    if use_lm:    
        protbert_path = "seq_models/huggingface/transformers/models--Rostlab--prot_bert"
        tokenizer = BertTokenizer.from_pretrained(protbert_path, do_lower_case=False )
        model = BertModel.from_pretrained(protbert_path)
        fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=device_id)

    number_of_proteins_with_no_graph = 0
    for item in tqdm(protein_list):
        lig_file_name = dataset_path + '/' + item + '/' + item + '_ligand.mol2'
        pocket_file_name = dataset_path + '/' + item + '/' + item + '_pocket.pdb'
        protein_path = dataset_path + '/' + item  + '/'
        if not os.path.exists(protein_path):
            print(f"protein path {protein_path} does not exist ")
            continue

        file_path = checkpoint_cache_path + '/' + item + '_graph.pkl' 
        # Check if the file exists
        if os.path.exists(file_path) and not no_from_cache: 
            with open(file_path, 'rb') as f:
                G = pickle.load(f)
                with open(file_path, 'wb') as f:
                    pickle.dump(G, f)
        else:
            score = res[item]
        
            lig_coord, lig_atom_fea, mol, h_num_lig, ligand_center, ligand_atom_feature_names = read_ligand(lig_file_name)
            pocket_coord, pocket_atom_fea,protein,h_num_pro,pro_seq, pocket_atom_feature_names= read_protein(pocket_file_name,fe if use_lm else None)  #,pro_seq

            if(mol != None) and (protein != None):
                # print(pro_seq)
                # get HeteroData started with the inter graph (ligand features, pocket features, interaction)
                G = compute_pocket_and_ligand_graph(lig_coord, pocket_coord, lig_atom_fea, pocket_atom_fea, score) # ligand nodes computed here. 
                l_edge_attr, l_edge_index = get_ligand_bonds(lig_atom_fea, mol, h_num_lig,score) # then compute ligand bonds 
                G["ligand", "lig_bond", "ligand"].edge_index = l_edge_index 
                G['ligand', 'lig_bond', 'ligand'].edge_attr = l_edge_attr

                # Some identifying information 

                G["pocket"].ft_names = pocket_atom_feature_names
                G["ligand"].ft_names = ligand_atom_feature_names 
                G.id = item 
                G.l_center = ligand_center 

        
            # Compute protein graph 
            protein_graph_items = load_protein_graph(protein_path,item)
            if protein_graph_items is not None:        
                X_ca, seq, node_s, node_v, edge_s, edge_v, edge_index = protein_graph_items 
                node_v = node_v.view(node_v.shape[0],-1) # 3,3 -> 9
                seq = seq.unsqueeze(1) # L -> L,1
                ft = torch.cat([node_s, node_v,seq], dim = -1) # N, (6+9+1=16)
                # Protein nodes
                G["protein"].pos = X_ca 
                G["protein"].ft = ft # ft is sequence, dihedrals, bb orientation vec, side chain vec 
            else: 
                number_of_proteins_with_no_graph += 1
                print(f"number of proteins with no graph: {number_of_proteins_with_no_graph}")     
            
            with open(file_path, 'wb') as f:
                pickle.dump(G, f)

        # Use all-atom to compute surface 
        # try: # this will throw an error if there's an atom not in ele2num 
        
        if all_atom: 
            file_path_aa = checkpoint_cache_path + '/' + item + '_aa.pkl' 
            if os.path.exists(file_path) and not no_from_cache: 
                with open(file_path, 'rb') as f:
                    Gaa = pickle.load(f)
                    with open(file_path, 'wb') as f:
                        pickle.dump(Gaa, f)
            else:
                protein_atom_coords, protein_atom_types, protein_atom_residue_indices = load_protein_atoms(protein_path, item)
                Gaa = HeteroData()
                Gaa["protein_all"].pos = torch.tensor(protein_atom_coords, dtype=torch.float32)
                Gaa["protein_all"].type = torch.from_numpy(protein_atom_types).long()
                Gaa["protein_all"].res = torch.tensor(protein_atom_residue_indices).long() 

                with open(file_path_aa, 'wb') as f:
                    pickle.dump(G,f)

        if G != None:
            G_list.append(G)
            if all_atom: 
                Gaa_list.append(Gaa)

    print('sample num: ', len(G_list))
    print('samples with all atom graphs', len(Gaa_list))
    print(f"number of proteins with no graph: {number_of_proteins_with_no_graph}")
    with open(processed_file, 'wb') as f:
        pickle.dump(G_list, f)
    with open(processed_file.replace(".pkl","_aa.pkl"),'wb') as f: 
        pickle.dump(Gaa_list,f)

    f.close()




if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description="split")
    parser.add_argument('--split', type=str, default = 'all')
    parser.add_argument('--train_general', action = 'store_true')
    parser.add_argument('--lm', action = 'store_true')
    parser.add_argument('--all_atom', action = 'store_true')
    parser.add_argument('--surface', action = 'store_true')
    args = parser.parse_args()


    # data paths 
    raw_data_path_test = 'data/refined-set' 
    raw_data_path_val = 'data/refined-set' 
    raw_data_path_train = 'data/refined-set' if not args.train_general else 'data/v2020-other-PL'

    # index files 
    index_path_test = 'data/CASF-2016/power_screening/CoreSet.dat' 
    index_path_val = 'data/index/INDEX_refined_data.2016'
    index_path_train = 'data/index/INDEX_refined_data.2016' if not args.train_general else 'data/v2020-other-PL/index/INDEX_general_PL_data.2020'

    # where to store 
    scope_folder = 'refined2016' if not args.train_general else 'general2020'
    type_folder = 'surface' if args.surface else 'all_atom' if args.all_atom else 'lm' if args.lm else 'basic' 
    out_folder = f'processed_data/{scope_folder}/{type_folder}'
    out_test = f'{out_folder}/test.pkl'
    out_val = f'{out_folder}/val.pkl'
    out_train = f'{out_folder}/train.pkl'
    os.makedirs(out_folder, exist_ok = True )
    
    # split 
    process_all = True if args.split == 'all' else False 
    split = args.split 
    
    # random seed 
    seed = 805
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # get list of complex names 
    protein_list_test = GetPDBList_core(Path = index_path_test)
    print('getting test index')
    res_test = GetPDBDict_core(Path = f'/homes/wongr/MFE/{index_path_test}')
  
    protein_list_val = GetPDBList(Path=index_path_val)
    protein_list_val = [i for i in protein_list_val if i not in protein_list_test]
    res = GetPDBDict(Path=index_path_val)
    res = {key: value for key, value in res.items() if key not in protein_list_test}
    idxs = np.random.choice(len(protein_list_val), size = int(0.1*(len(protein_list_val)-len(protein_list_test))), replace = False) # 4057 for refined, 285 for core
    protein_list_val = [protein_list_val[idx] for idx in idxs]
    res_val = {key: res[key] for key in protein_list_val}
    

    protein_list_train = GetPDBList(Path=index_path_train) 
    protein_list_train = [i for i in protein_list_train if i not in protein_list_val and i not in protein_list_test]
    res_train = GetPDBDict(Path=index_path_train)
    res_train = {key: value for key, value in res_train.items() if key not in protein_list_val and key not in protein_list_test}
    # res_train = {'5ylv': res_train['5ylv']}
    # if num_train is not None: 
    #     idxs = np.random.choice(len(protein_list_train), size = num_train, replace = False)
    #     protein_list_train = [protein_list_train[idx] for idx in idxs]
    #     res_train = {key: res_train[key] for key in protein_list_train}
    #     out_train = f'data/cache/train_{num_train}.pkl'


    overlap = list(set(protein_list_train) & set(protein_list_val))
    print(f"check if overlap {overlap}")

    cache_path = f"{out_folder}/cache"
    os.makedirs(cache_path, exist_ok = True )


    if process_all: 
        print("processing all")
        process_raw_data(raw_data_path_test,res_test, out_test, protein_list_test, use_lm = args.lm, all_atom=args.all_atom,checkpoint_cache_path = cache_path)
        process_raw_data(raw_data_path_val,res_val, out_val, protein_list_val, use_lm = args.lm, all_atom=args.all_atom,checkpoint_cache_path = cache_path)
        process_raw_data(raw_data_path_train,res_train, out_train, protein_list_train, use_lm = args.lm, all_atom=args.all_atom,checkpoint_cache_path = cache_path)
       
        
    else: 
        if split == 'train':   
            process_raw_data(raw_data_path_train,res_train, out_train, protein_list_train,use_lm = args.lm, all_atom=args.all_atom, checkpoint_cache_path = cache_path)
    
        elif split == 'val':  
            process_raw_data(raw_data_path_val,res_val, out_val, protein_list_val,use_lm = args.lm, all_atom=args.all_atom,checkpoint_cache_path = cache_path)
        
        elif split == 'test': 
            process_raw_data(raw_data_path_test,res_test, out_test, protein_list_test, no_from_cache= True,use_lm = args.lm, all_atom=args.all_atom,checkpoint_cache_path = cache_path)
        











