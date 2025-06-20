import os
import torch
import warnings
from sklearn import metrics
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from torch_geometric.data import DataLoader
from torch_geometric.data import HeteroData
from sklearn.linear_model import LinearRegression
from easydict import EasyDict
import scipy
import yaml
import wandb
from datetime import datetime


from train_data_location import train_path, val_path, test_path 
from net_transform_graphs_ import  *
from net_logic import field,schedule,l_schedule, args, edge_sets 
from net_analysis import TrackOutput, TrackReps, DatasetStatistics

import pickle 
from tqdm import tqdm 
from torch_geometric.data import Batch

import time
from torch.utils.data import Subset, RandomSampler,SubsetRandomSampler
import random 
import numpy as np 
import yaml

from net import Net 

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
g = torch.Generator()
g.manual_seed(seed)

        
def metrics_reg(targets,predicts):
    mae = metrics.mean_absolute_error(y_true=targets,y_pred=predicts)
    # rmse = metrics.mean_squared_error(y_true=targets,y_pred=predicts,squared=False)
    rmse = metrics.root_mean_squared_error(y_true=targets,y_pred=predicts)
    r = scipy.stats.mstats.pearsonr(targets, predicts)[0]

    x = [ [item] for item in predicts]
    lr = LinearRegression()
    lr.fit(X=x,y=targets)
    y_ = lr.predict(x)
    sd = (((targets - y_) ** 2).sum() / (len(targets) - 1)) ** 0.5

    return [mae,rmse,r,sd]


def train(train_loader, val_loader, test_loader, kf_filepath, model, config=None):

    print('start training')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience=5, verbose = True)

    # initialize analysis 
    analyze = TrackOutput(kf_filepath)
    rep_tracker = TrackReps(kf_filepath)

    loss_list = []
    best_mae = float('inf')
    best_rmse = float('inf')
    
    logs = {}
  

    for epoch in range(args.epochs):
        model.train()
        loss_epoch = 0
        n = 0
        print(f"epoch {epoch}")
        for batch, data in enumerate(tqdm(train_loader)):
            torch.autograd.set_detect_anomaly(True)
        

            s=time.time() 
            e=time.time() 
            optimizer.zero_grad()
            s=time.time() 
            out = model(data) # out is affinity
            e=time.time() 
            # print(f'forward {e-s}s')
            # print(f'computing mse on {data["ligand"].score.shape}')
            # loss = F.huber_loss(out, data["ligand"].score, delta = args.huber_delta, reduction = 'mean')+args.l1_reg_weight*torch.mean(torch.abs(out))
            loss = F.mse_loss(out, data["ligand"].score)+args.l1_reg_weight*torch.mean(torch.abs(out))
            loss_epoch += loss.item()

            if args.forward_debug: 
                exit() 

            s=time.time()
            loss.backward()
            e=time.time() 
            print(f'backward {e-s}s')
            optimizer.step()
            n += 1

        loss_list.append(loss_epoch / n)
        
    
        # wandb.log(logs, step=epoch + 1)

        

        print('epoch:', epoch, ' loss:', loss_epoch / n)

        val_err = my_val(model, val_loader, device, scheduler, epoch)
        val_mae = val_err[0]
        val_rmse = val_err[1]

        if val_rmse < best_rmse and val_mae < best_mae:
            print('********save model*********')
            torch.save(model.state_dict(), kf_filepath+'/best_model.pt')
            best_mae = val_mae
            best_rmse = val_rmse

            f_log = open(file=(kf_filepath+"/metrics_log.txt"), mode="a")
            str_log = 'epoch: '+ str(epoch) + ' val_mae: ' + str(val_mae) + ' val_rmse: ' + str(val_rmse)+ '\n'
            f_log.write(str_log)
            f_log.close()
            print("running test")
            test_err = my_test(test_loader, kf_filepath, config, analyze, rep_tracker)

            # log wandb (log everything once, with epoch) 
            wandb.log({'epoch': epoch, 'training loss': loss_epoch/n, 'learning rate': optimizer.param_groups[0]['lr'], 
                   'val loss': loss_epoch / n, 'val mae': val_err[0],'val rmse': val_err[1], 'val r': val_err[2],'val sd': val_err[3],
                   'test mae': test_err[0],'test rmse': test_err[1],'test r': test_err[2],'test sd': test_err[3]})
    
        else: 
            # log wandb (log everything once, with epoch) 
            wandb.log({'epoch': epoch, 'training loss': loss_epoch/n, 'learning rate': optimizer.param_groups[0]['lr'], 
                    'val loss': loss_epoch / n, 'val mae': val_err[0],'val rmse': val_err[1], 'val r': val_err[2],'val sd': val_err[3]})

      

    plt.plot(loss_list)
    plt.ylabel('Loss')
    plt.xlabel("time")
    plt.savefig(kf_filepath+'/loss.png')
    plt.show() 
    
    analyze.save_performance() 
    analyze.plot_error_distribution_over_time()
    analyze.plot_pred_vs_true_over_time()


def my_val(model, val_loader, device, scheduler, epoch):
    p_affinity = []
    y_affinity = []

    model.eval()
    loss_epoch = 0
    n = 0
    for data in tqdm(val_loader):
        with torch.no_grad():
            # data = data.to(device)
            predict = model(data)
            loss =  F.mse_loss(predict, data["ligand"].score)
            print(f"val loss {loss.item()}")
            loss_epoch += loss.item()
            n += 1

            print(f"val pred-score {predict-data['ligand'].score}")
            print(f"val pred: {predict}")
            print(F"val ground truth: {data['ligand'].score}")

            p_affinity.extend(predict.view(-1).cpu().tolist())
            y_affinity.extend(data["ligand"].score.view(-1).cpu().tolist())


    scheduler.step(loss_epoch / n)
    affinity_err = metrics_reg(targets=y_affinity,predicts=p_affinity)
    
    return affinity_err


def my_test(test_loader, kf_filepath, config, analyze=None, rep_tracker=None):
    p_affinity = []
    y_affinity = []
    ids = [] 
    

    m_state_dict = torch.load(kf_filepath+'/best_model.pt')
    best_model = Net(args,field,schedule,l_schedule, edge_sets).to(device)
    best_model.inspect = True 
    best_model.load_state_dict(m_state_dict)
    best_model.eval()

    for i, data in enumerate(test_loader, 0):
        with torch.no_grad():
            # print("test number ", i)
            # data = data.to(device)
            predict = best_model(data)
            # print(f"test pred-score: {predict-data['ligand'].score}")
            # print(f"test pred: {predict}")
            # print(F"test ground truth: {data['ligand'].score}")
            p_affinity.extend(predict.view(-1).cpu().tolist())
            y_affinity.extend(data["ligand"].score.view(-1).cpu().tolist())
            ids.extend(data.id)

            # store activations and label with "epoch "
            rep_tracker.add_(best_model.layer_outputs)

    
    #### LOG AND PLOT ERRORS AND REPS 
    # add completed list of preds, trues and errors. Then update distribution of errors, preds and trues over time. 
    print(f"prediction tensor is {len(p_affinity)}")
    analyze.add_performance(p_affinity, y_affinity, [abs(p_affinity[i]-y_affinity[i]) for i in range(len(p_affinity))],ids)
    analyze.plot_error_distribution_over_time()
    analyze.plot_pred_vs_true_over_time()
    rep_tracker.finish_epoch() 
    rep_tracker.analyze()


    affinity_err = metrics_reg(targets=y_affinity,predicts=p_affinity)

    f_log = open(file=(kf_filepath+"/metrics_log.txt"), mode="a")
    str_log = 'mae: '+ str(affinity_err[0]) + ' rmse: '+ str(affinity_err[1]) + ' r: '+ str(affinity_err[2]) +' sd: '+ str(affinity_err[3]) + '\n'
    f_log.write(str_log)
    f_log.close()

    return affinity_err 
    



if __name__ == '__main__':
  
    """ Please run process.py to preprocess the raw data and set up the training, validation, and test sets """
    
    # how many complexes, how many per batch, where to write model to. 
    limit_complexes = args.limit_complexes  # 8000  # just for train, set to 0 for full (~5-7 min to load)
    batch_size = args.batch_size
    device = torch.device(f'cuda:{args.cuda_id}')
    print(f"device {device}")
    mode = args.mode 

    # name the experiment 
    if args.name == None: 
        date = datetime.today().strftime('%Y-%m-%d')
        # complex_params = f'ld{args.ld}pd{args.pd}pld{args.pld}pcd{args.pcd}lcd{args.lcd}_rmaxnb{args.ca_max_nb}_plint{not args.no_pl_int}_lpint{not args.no_lp_int}'
        # nn_params = f'lr{args.learning_rate}_wdecay{args.weight_decay}_ns{args.ns}_nv{args.nv}_intlayers{args.int_layers}_bn{not args.no_bn}'
        exp_name = f"{date}_{mode}"
        print(f"Default experiment name is {exp_name}")
    else: 
        exp_name = args.name 
        print(f"Experiment name is {exp_name}")
    print(f"Experiment name is {exp_name}")
    filepath_base = f'./trained_models/{exp_name}'
    filepath = filepath_base 
    index = 0 
    if os.path.exists(filepath_base):
        while os.path.exists(filepath):
            index += 1
            filepath = f"{filepath_base}_{index}"
        exp_name = f"{exp_name}_{index}"
            
        os.makedirs(filepath, exist_ok = True) 
    elif not os.path.exists(filepath_base):
        filepath = filepath_base 
        os.makedirs(filepath, exist_ok = True)
   
    
    # write parameter file into yaml 
    yaml_path = os.path.join(filepath, "config.yaml")
    with open(yaml_path,"w") as f:
        yaml.dump(args,f, default_flow_style=False)
    

    print("Loading preprocessed datasets...")
    # Load the preprocessed datasets, with train based on limit_complexes.
    start_time = time.time() 
    # with open("data/cache/val_final.pkl", "rb") as f: 
    with open(val_path, "rb") as f: 
        val_dataset = pickle.load(f)
    print("loaded val")
    with open(test_path, "rb") as f: 
        test_dataset = pickle.load(f)
    print("loaded test")
    if limit_complexes !=0:
        train_subset_path = f"data/cache/train_final_limit{limit_complexes}.pkl"
        if os.path.exists(train_subset_path):
            print("loading train subset")
            with open(train_subset_path, "rb") as f:
                train_dataset = pickle.load(f)
                print("len subset", len(train_dataset))
        else: 
            with open(train_path, "rb") as f:
                print("loading train")
                train_dataset = pickle.load(f)
            # idx = np.asarray(random.sample(range(len(train_dataset)), limit_complexes))
            idx = np.asarray(random.sample(range(len(train_dataset)), limit_complexes))
            train_subset = Subset(train_dataset, idx)
            train_subset = [i for i in train_subset]
            print("len subset", len(train_subset))
            with open(train_subset_path, "wb") as f:
                pickle.dump(train_subset, f)
                train_dataset = train_subset 
    else: 
        with open(train_path, "rb") as f: 
            train_dataset = pickle.load(f)
            
    # timing dataset loading 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Datasets loaded in {elapsed_time:.2f} seconds.")

    # define model 
    model = Net(args,field,schedule,l_schedule, edge_sets)#, inspect=True)
    model = model.to(device)

    # apply graph tranformations 
    transform = Transform(args)
    print('applying transformations to graph')

    # print number of parameters
    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    # wandb 
    if not args.forward_debug: 
        wandb.init(
            entity='wongrp',
            settings=wandb.Settings(start_method="fork"),
            project='mp_dta',
            name= exp_name,
            group=args.wandb_group,
            job_type=args.wandb_jobtype,
            config=args
        )
        wandb.log({'numel': numel}) # log number of parameters 
    


        # Upload YAML file and Python scripts to wandb as artifacts
        wandb.save("net.py")  # Upload the model definition script
        wandb.save("net_logic.py")  
        wandb.save("args.py") 
        wandb.save("net_convs.py")
        wandb.save("net_fusion.py")
        wandb.save("net_operations.py")
        wandb.save("net_l_schedule.py")
        wandb.save("train.py")


    
    _original_inc = HeteroData.__inc__

    def debug_inc(self, key, value, store):
        try:
            return _original_inc(self, key, value, store)
        except Exception as e:
            print(f"__inc__ failed on key: {key}")
            print(f"Value: {value}")
            print(f"Store type: {type(store)}")
            print("Store contents:")
            for k, v in store.items():
                print(f"  {k}: type={type(v)}, value={v if isinstance(v, (int, float, str)) else 'tensor'}")
            raise

    HeteroData.__inc__ = debug_inc

    train_dataset = CachedTransformDataset(train_dataset, transform.build_multiplex,device =device)
    val_dataset = CachedTransformDataset(val_dataset, transform.build_multiplex, device =device)
    test_dataset = CachedTransformDataset(test_dataset, transform.build_multiplex, device=device)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last = True)


    train_ids = [d.id for d in train_dataset]
    test_ids = [d.id for d in test_dataset] 
    val_ids = [d.id for d in val_dataset]
 
    print(f"checking train/test overlap {list(set(train_ids) & set(test_ids) )}")
    print(f"checking val/test overlap {list(set(val_ids) & set(test_ids) )}")
        # test set 193%32=1 
        # set shuffle to True here with set seed to retrieve ops, then shuffle = False later on to maintain this order. 
        

    # statistics
    train_stats = DatasetStatistics(dataset=train_dataset, save_dir=f"{filepath}/statistics", prefix="train")
    val_stats = DatasetStatistics(dataset=val_dataset, save_dir=f"{filepath}/statistics", prefix="val")
    test_stats = DatasetStatistics(dataset=test_dataset, save_dir=f"{filepath}/statistics", prefix="test")
    train_stats.compute()
    val_stats.compute()
    test_stats.compute()


    # train 
    ti = time.time()
    train(train_loader, val_loader, test_loader, filepath, model)
    tf = time.time()
    print(f"trained in {tf-ti}s")

