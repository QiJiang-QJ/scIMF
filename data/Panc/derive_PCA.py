import pandas as pd
import numpy as np
import natsort
import scanpy
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import dump

leaveout_type = ["leaveout1", "leaveout2", "leaveout3", "leaveout4", "leaveout5", "leaveout6", "leaveout7"][6]
test_tpsS_dict = {"leaveout1": [1], "leaveout2":[2], "leaveout3":[3], "leaveout4":[4], "leaveout5":[5], "leaveout6":[6], "leaveout7":[7]}
test_tps = test_tpsS_dict[leaveout_type]
n_tps=8
all_tps = list(range(n_tps)) 
train_tps = list(sorted(set(all_tps)-set(test_tps))) 


cnt_data = pd.read_csv("{}-count_data-hvg.csv".format(leaveout_type), header=0, index_col=0)
meta_data = pd.read_csv("meta_data.csv",header=0, index_col=0)
meta_data = meta_data.loc[cnt_data.index,:]
cell_stage = meta_data["CellWeek"]
unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
cell_tp = np.zeros((len(cell_stage), ))
cell_tp[cell_tp == 0] = np.nan
for idx, s in enumerate(unique_cell_stages):
    cell_tp[np.where(cell_stage == s)[0]] = idx
meta_data["tp"] = cell_tp
ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)

ann_data.X = ann_data.X.astype(float)
scanpy.pp.normalize_per_cell(  # normalize with total UMI count per cell
    ann_data, key_n_counts='n_counts_all', counts_per_cell_after=1e4
)
scanpy.pp.log1p(ann_data)

data = ann_data.X 
data_listAllT = [torch.FloatTensor(data[np.where(cell_tp == t)[0], :]) for t in range(0, n_tps)]  # (# tps, # cells, # genes)

train_list = [data_listAllT[t] for t in train_tps]
test_list = [data_listAllT[t] for t in test_tps]

scaler = StandardScaler()
pca_model = PCA(n_components=50, svd_solver="arpack")
train_concat = np.concatenate(train_list, axis=0)
train_scale_concat = scaler.fit_transform(train_concat)
xp_train_concat = pca_model.fit_transform(train_scale_concat)
xp_all = pca_model.transform(scaler.transform(data))

data_dir = '{}-PCA_data.csv'.format(leaveout_type)
xp_all = pd.DataFrame(xp_all, index = cnt_data.index)
xp_all.to_csv(data_dir, index=True)
print(f"PCA data have been saved to {data_dir}")


PCA_path = "{}-PCA_model.joblib".format(leaveout_type)
PCA_models_dict = {
    "scaler": scaler,
    "pca": pca_model
}
dump(PCA_models_dict, PCA_path)
print(f"PCA Models have been saved to {PCA_path}")

