import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import os
import torch
import json
import pandas as pd
from collections import defaultdict
import os
import torch
import json
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

"""
TrackError 
DatatsetStatistics 
TrackReps
"""

class DatasetStatistics:
    def __init__(self, dataset, save_dir, prefix="train"):
        self.dataset = dataset  # will call dataset[i], activating __getitem__
        self.save_dir = save_dir
        self.prefix = prefix
        os.makedirs(self.save_dir, exist_ok=True)

        self.node_accum = defaultdict(lambda: defaultdict(float))
        self.edge_accum = defaultdict(lambda: defaultdict(float))
        self.node_counts = defaultdict(int)
        self.edge_counts = defaultdict(int)
        self.result_rows = []

    def compute(self):
        for i in range(len(self.dataset)):
            data = self.dataset[i]  # activates get_item and transforms
            self.update_from_graph(data)
        self.finalize()

    def update_from_graph(self, data):
        for node_type in data.node_types:
            if node_type.startswith("virtual") or 'ft' not in data[node_type]:
                continue
            ft = data[node_type].ft
            self.node_accum[node_type]['sum'] += ft.sum(dim=0)
            self.node_accum[node_type]['sum_sq'] += (ft ** 2).sum(dim=0)
            self.node_accum[node_type]['min'] = (
                ft.min(dim=0).values if 'min' not in self.node_accum[node_type]
                else torch.minimum(self.node_accum[node_type]['min'], ft.min(dim=0).values)
            )
            self.node_accum[node_type]['max'] = (
                ft.max(dim=0).values if 'max' not in self.node_accum[node_type]
                else torch.maximum(self.node_accum[node_type]['max'], ft.max(dim=0).values)
            )
            self.node_counts[node_type] += ft.size(0)

        for edge_type in data.edge_types:
            if 'edge_attr' not in data[edge_type]:
                continue
            ea = data[edge_type].edge_attr
            self.edge_accum[edge_type]['sum'] += ea.sum(dim=0)
            self.edge_accum[edge_type]['sum_sq'] += (ea ** 2).sum(dim=0)
            self.edge_accum[edge_type]['min'] = (
                ea.min(dim=0).values if 'min' not in self.edge_accum[edge_type]
                else torch.minimum(self.edge_accum[edge_type]['min'], ea.min(dim=0).values)
            )
            self.edge_accum[edge_type]['max'] = (
                ea.max(dim=0).values if 'max' not in self.edge_accum[edge_type]
                else torch.maximum(self.edge_accum[edge_type]['max'], ea.max(dim=0).values)
            )
            self.edge_counts[edge_type] += ea.size(0)

    def finalize(self):
        for node_type in self.node_accum:
            count = self.node_counts[node_type]
            mean = self.node_accum[node_type]['sum'] / count
            std = ((self.node_accum[node_type]['sum_sq'] / count) - mean ** 2).sqrt()
            self.result_rows.append({
                "type": "node",
                "name": node_type,
                "mean": mean.mean().item(),
                "std": std.mean().item(),
                "min": self.node_accum[node_type]['min'].min().item(),
                "max": self.node_accum[node_type]['max'].max().item(),
                "count": count
            })

        for edge_type in self.edge_accum:
            count = self.edge_counts[edge_type]
            mean = self.edge_accum[edge_type]['sum'] / count
            std = ((self.edge_accum[edge_type]['sum_sq'] / count) - mean ** 2).sqrt()
            self.result_rows.append({
                "type": "edge",
                "name": str(edge_type),
                "mean": mean.mean().item(),
                "std": std.mean().item(),
                "min": self.edge_accum[edge_type]['min'].min().item(),
                "max": self.edge_accum[edge_type]['max'].max().item(),
                "count": count
            })


        df = pd.DataFrame(self.result_rows)
        csv_path = os.path.join(self.save_dir, f"{self.prefix}_feature_statistics.csv")
        json_path = os.path.join(self.save_dir, f"{self.prefix}_feature_statistics.json")

        df.to_csv(csv_path, index=False)
        with open(json_path, 'w') as f:
            json.dump(self.result_rows, f, indent=2)

        print(f"Saved dataset statistics to:\n  {csv_path}\n  {json_path}")


class TrackOutput:
    def __init__(self, save_dir,split = 'test'):
        self.save_dir = save_dir
        self.basedir = f'figs/error_progress/{split}'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, f"{self.basedir}/per_example"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, f"{self.basedir}/scatter_error_per_epoch"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, f"{self.basedir}/scatter_preds_per_epoch"), exist_ok=True)


        self.outs = []
        self.scores = []
        self.ae = []
        self.ids = []

        # Dictionary to track predictions over time for each ID
        self.pred_history = {}  # id -> list of predictions
        self.true_history = {}  # id -> list of ground-truth scores
        self.ae_history = {}    # id -> list of absolute errors


    # def get_layer_outputs(ouput_dict): 
        
    def add_performance(self, out, score, error, ids):
        # print(f"adding list of length {len(out)} to list of length {len(self.outs)}")
        # print(f"adding {out} to {self.outs}")
        self.outs.append(out)
        self.scores.append(score)
        self.ae.append(error)
        # print(f"ae length {len(self.ae)} ad shape {self.ae[0].shape}")
        # print(f"list of ids is {ids}")
        for idx,id in enumerate(ids): 
       
            self.ids.append(id)

            # Update history tracking
            if id not in self.pred_history:
                self.pred_history[id] = []
                self.true_history[id] = []
                self.ae_history[id] = []

            self.pred_history[id].append(out[idx])
            self.true_history[id].append(score[idx])
            self.ae_history[id].append(error[idx])


            # Live plot of performance evolution
            self.plot_id_progress(id)


    def plot_id_progress(self, id):
        fig_path = os.path.join(self.save_dir, f"{self.basedir}/per_example", f"{id}_progress.png")
        plt.figure()
        plt.plot(self.true_history[id], label='True', marker='o')
        plt.plot(self.pred_history[id], label='Predicted', marker='x')
        plt.plot(self.ae_history[id], label='Abs Error', linestyle='--')
        plt.title(f"Performance for: {id}")
        plt.xlabel("Checkpoint Index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

    def save_performance(self):
        num_examples = 30
        ae = np.array(self.ae)
        scores = np.array(self.scores)
        outs = np.array(self.outs)
        ids = np.array(self.ids)


        best_idx = np.argsort(ae)[:num_examples]
        worst_idx = np.argsort(ae)[-num_examples:]

        def collect(idx):
            return {
                'id': ids[idx].tolist(),
                'ae': ae[idx].tolist(),
                'score': scores[idx].tolist(),
                'pred': outs[idx].tolist()
            }

        best = collect(best_idx)
        worst = collect(worst_idx)

        best_df = pd.DataFrame(best)
        worst_df = pd.DataFrame(worst)

        best_df.to_csv(os.path.join(self.save_dir, "best_preds.csv"), index=False)
        worst_df.to_csv(os.path.join(self.save_dir, "worst_preds.csv"), index=False)
        best_df.to_csv(os.path.join(self.save_dir, "figs", "best_preds.csv"), index=False)
        worst_df.to_csv(os.path.join(self.save_dir, "figs", "worst_preds.csv"), index=False)

        # Histogram of AE
        plt.figure()
        plt.hist(ae, bins=10)
        plt.xlabel("Absolute Error")
        plt.ylabel("Frequency")
        plt.title("Absolute Error Histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "figs", "aehist.png"))
        plt.close()

        # Histogram of predicted and true scores
        plt.figure()
        plt.hist(outs, bins=10, alpha=0.8, label='Predicted')
        plt.hist(scores, bins=10, alpha=0.6, label='True')
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.title("Score Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "figs", "scorehist.png"))
        plt.close()


    def plot_pred_vs_true_over_time(self):
        for epoch, (preds, trues) in enumerate(zip(self.outs, self.scores)):
            print(f"plotting epoch {epoch}, prediction is {len(preds)}")
            preds = np.array(preds)
            trues = np.array(trues)

            plt.figure(figsize=(6, 6))
            plt.scatter(trues, preds, alpha=0.5, s=10)
            plt.plot([trues.min(), trues.max()], [trues.min(), trues.max()], 'k--', lw=1)
            plt.xlabel("Ground Truth")
            plt.ylabel("Prediction")
            plt.title(f"Epoch {epoch}: Prediction vs Ground Truth")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"{self.basedir}/scatter_preds_per_epoch/epoch_{epoch}.png"))
            plt.close()


    def plot_error_distribution_over_time(self):
        for epoch, preds in enumerate(self.outs):
            errors = np.abs(np.array(preds) - np.array(self.scores[epoch]))

            plt.figure()
            plt.hist(errors, bins=30, alpha=0.7)
            plt.title(f"Epoch {epoch} Error Distribution")
            plt.xlabel("Absolute Error")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"{self.basedir}/scatter_error_per_epoch/epoch_{epoch}.png"))
            plt.close()




class TrackReps:
    def __init__(self, save_dir, split='test'):
        self.save_dir = save_dir
        self.basedir = f"figs/{split}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, f"{self.basedir}/rep_statistics"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, f"{self.basedir}/rep_knn"), exist_ok=True)
        self.rep_storage = {}  # tag -> list of (tensor, epoch)
        self.epoch = 1 
        self.epoch_done = True 
        self.scores = []
        self.outs = [] 
        self.ae = [] 
        self.ids = [] 
        

    def add(self, tensor, score, tag, epoch):
        if tag not in self.rep_storage:
            self.rep_storage[tag] = []
        self.rep_storage[tag].append((tensor.detach().cpu(), epoch))
        

    def finish_epoch(self): 
        self.epoch += 1 
        self.epoch_done = True 
    

    def add_(self,layer_outputs):

        for layer_output in layer_outputs: 
            
            tag = layer_output['name']

            if tag not in self.rep_storage:
                self.rep_storage[tag] = []
            epoch = self.epoch
            tensor = layer_output['ft']

            if self.epoch_done: 
                self.rep_storage[tag].append(tensor.detach().cpu())
            else: 
                self.rep_storage[tag][-1] = torch.cat([self.rep_storage[tag][-1],tensor.detach().cpu()],dim=0)
        
        if self.epoch_done: 
            self.epoch_done = False   

    def add_performance(self, out, score, error,id):
        self.outs.append(out)
        self.ae.append(error)  
        if len(self.scores)==0:
            self.scores = np.asarray(score)
            self.ids = id 

    def knn_on_vectors(self, vecs, scores,k=5):  
        # vectors: (N,D) (numpy array or torch.Tensor.cpu().numpy())
        # scores: shape (N) (numpy array)
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean').fit(vecs)
        distances, indices = nbrs.kneighbors(vecs)
        print(f"indices {indices}")
        neighbors = scores[indices[:,1:]] # skip the first neighbor (it's the point itself)

        return neighbors

    def plot_knn(self, vecs, scores, tag, epoch):
        neighbors= self.knn_on_vectors(vecs, scores)
        std_per_point = np.std(neighbors, axis=1)
        mean_abs_diff_per_point = np.mean(np.abs(neighbors - scores[:, None]), axis=1)

        # Visualize: local smoothness vs score
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(scores, std_per_point, alpha=0.5)
        plt.xlabel("Score")
        plt.ylabel("Std of Neighbor Scores")
        plt.title("Local Score Variability")

        plt.subplot(1, 2, 2)
        plt.scatter(scores, mean_abs_diff_per_point, alpha=0.5)
        plt.xlabel("Score")
        plt.ylabel("Mean | Neighbor Score - Own Score |")
        plt.title("Local Mean Absolute Difference")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{self.basedir}/rep_knn/{tag}_knn_epoch{epoch}.png"))
        plt.close()


    def analyze(self):
        for tag, reps in self.rep_storage.items():
            norms_by_epoch = []
            var_by_epoch = []

            for epoch, tensor in enumerate(reps):
                print(f"we've stored {len(self.scores)} scores")
                print(f"which we will associate with {tensor.shape} hidden vectors")
                if tensor.shape[0] == len(self.scores):
                    self.plot_knn(tensor, self.scores, tag, epoch)
                
                norms = tensor.norm(dim=-1).numpy()
                feature_mean = tensor.mean(dim=0).numpy()
                feature_std = tensor.std(dim=0).numpy()

                norms_by_epoch.append((epoch, norms))
                var_by_epoch.append((epoch, feature_std))

                # Plot norm histogram
                plt.figure()
                plt.hist(norms, bins=30)
                plt.title(f"Node-wise Norm Distribution [{tag}], Epoch {epoch}")
                plt.xlabel("||h||")
                plt.ylabel("# Nodes")
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, f"{self.basedir}/rep_statistics/{tag}_norm_epoch{epoch}.png"))
                plt.close()

                # Plot feature std
                plt.figure()
                plt.plot(feature_std)
                plt.title(f"Per-Feature Std [{tag}], Epoch {epoch}")
                plt.xlabel("Feature Index")
                plt.ylabel("Std")
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, f"{self.basedir}/rep_statistics/{tag}_featurestd_epoch{epoch}.png"))
                plt.close()

            # Optional: track evolution over time
            epochs, norms_list = zip(*norms_by_epoch)
            avg_norms = [np.mean(norms) for norms in norms_list]

            plt.figure()
            plt.plot(epochs, avg_norms)
            plt.title(f"Average Norm Over Epochs [{tag}]")
            plt.xlabel("Epoch")
            plt.ylabel("Avg ||h||")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"figs/rep_statistics/{tag}_avg_norms_over_time.png"))
            plt.close()

            # You can also add PCA spectrum, participation ratio, or CKA here
