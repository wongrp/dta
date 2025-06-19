import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    #  architecture 
    parser.add_argument('--mode', type=str, default='plc',
                    choices=['plc', 'rlx', 'rlp', 'rl'],
                    help='Choose graph structure and schedule.')
    
    # WANDB experiment 
    parser.add_argument('--name', type=str, default = None)
    parser.add_argument('--wandb_group', type=str, default= None)
    parser.add_argument('--wandb_jobtype', type=str, default = None)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--huber_delta', type=float, default = 1000)

    # Encoder 
    parser.add_argument('--din_edge', type=int, default = 32)
    # Regularization 
    parser.add_argument('--dropout', type=float, default = 0.5)
    parser.add_argument('--dropout_mp', type=float, default = 0.3)
    parser.add_argument('--dropout_mpreadout', type=float, default = 0)
    parser.add_argument('--dropout_edge_emb', type=float, default=0.3)
    parser.add_argument('--dropout_edge', type=float, default=0.3)
    parser.add_argument('--no_bn', action='store_true') 
    parser.add_argument('--bn_mpr', action='store_true', help = 'add a batch norm at mp readout. default false.')
    parser.add_argument('--bn_mplayer', action = 'store_true', help = 'add a batch norm at the end of each mp layer' )
    parser.add_argument('--bn_edge_emb', action ='store_true')
    parser.add_argument('--l1_reg_weight', type = float, default = 0)
    parser.add_argument('--weight_decay', type = float, default = 1e-3)
                         
    # Study 
    parser.add_argument('--forward_debug', action='store_true')
    parser.add_argument('--inspect_tensors', action ='store_true')

    # Setup 
    parser.add_argument('--device', type=str, default='cuda', help='set device, like cuda or cpu')
    parser.add_argument('--cuda_id',type=int, default = 0)

    # Data 
    parser.add_argument('--limit_complexes', type=int, default=0, help='set to nonzero value for number of training examples.')
    parser.add_argument('--lsh', type=int, default=2)

    # Equivariant MPNN
    # default params are updated to current best run  
    parser.add_argument('--no_res',action='store_true')
    parser.add_argument('--pkaa', action = 'store_true')
    parser.add_argument('--ns', type=int, default=18, help = "scalars passed as messages")
    parser.add_argument('--nv', type=int, default=16, help = "l=1 (or l=2) vecs passed as messages")
    parser.add_argument('--layers', type = int, default = 3, help = "number of equivariant interaction layers" )
    
    parser.add_argument('--ld', type=int, default = 5)
    parser.add_argument('--rd', type=int, default = 15) # redundant with pd
    parser.add_argument('--rld', type=int, default = 15) # redundant
    parser.add_argument('--rcd', type=int, default = 100)
    parser.add_argument('--lcd', type=int, default = 60)
    parser.add_argument('--rlcd', type=int,default = 100)
    parser.add_argument('--plcd', type=int, default = 15)
    parser.add_argument('--pd', type=int,default = 7)
    parser.add_argument('--pcd', type=int,default = 15)
    parser.add_argument('--pld', type=float,default = 5)
    parser.add_argument('--rpd', type = float, default = 5)
    parser.add_argument('--rpcd', type= float, default = 5)

    parser.add_argument('--r_max_nb', type=int, default = 12)
    parser.add_argument('--p_max_nb', type=int, default = 8)
    parser.add_argument('--l_max_nb', type = int, default = 8)
    parser.add_argument('--no_pl_int', action='store_true')
    parser.add_argument('--no_lp_int', action='store_true')

    # Readout 
    parser.add_argument('--ns_readout', type=int, default=16)
    parser.add_argument('--readout_dim', type=int, default=16)
    parser.add_argument('--use_inter_readout', action = 'store_true')
    parser.add_argument('--fusion_type', type=str, default='MLP',
                        choices=['MLP', 'add_sub', 'gate', 'cross_att', 'transformer'],
                        help='Type of fusion module to use.')


    # DiffusionNet 
    parser.add_argument('--num_eig', type=int, default=128, help='eigenvectors of laplacian')
    parser.add_argument('--norm', type=bool, default=False, help='normalize surface points')
    parser.add_argument('--save_frames', type=bool, default=False, help='save surface xyz frames')
    parser.add_argument('--surface_type', type=str, default="both",choices = ["both", "pocket_surface","surface"])
    
    # Archictectural components 
    # parser.add_argument(--, type = str, default = ) # linear / nonlienear embeddings 
    # parser.add_argument(--, type = str, default = ) # batch
    # parser.add_argument(--, type = str, default = )
    # parser.add_argument(--, type = str, default = )
    # parser.add_argument(--, type = str, default = ) 




    # Parse arguments
    args = parser.parse_args()

    return args 