"""
inspects dataset for overlaps b/w val and test set and removes them 
"""

from torch_geometric.loader import DataLoader
import pickle
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm
from train_data_location import train_path, test_path, val_path

torch.set_printoptions(threshold=float('inf'), linewidth=200)

with open(test_path, "rb") as f: 
    test_dataset = pickle.load(f)
with open(train_path, "rb") as f: 
    train_dataset = pickle.load(f)

batch_size = 1
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = True)
train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=True )


test_ids = [d.id for d in test_dataset]
train_dataset = [d for d in train_dataset if d.id not in test_ids]
with open(train_path, "wb") as f:
    pickle.dump(train_dataset, f)