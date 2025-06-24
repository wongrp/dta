Create an environment with
```
conda create --name dta python=3.10
conda activate dta

pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

pip install -r requirements.txt 
```

To run:

```
python train.py
```

By default this trains a network that takes pocket (p), ligand (l) and complex (c) objects as input. Other objects include the combined residue-ligand object (x) and the combined residue-pocket object (z) You can change that with the tags

```
--mode rlp 
--mode rlx
--mode rl 
--mode plc
--mode prlz (not implemented yet)
```
During and after training you can go to `trained_models/date_mode` to check progress, including per-example error over time, prediction scatter plots over time, and outputs or hidden layer activations on the test set.
