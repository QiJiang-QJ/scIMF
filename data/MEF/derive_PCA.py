import pandas as pd
import numpy as np
import natsort
import scanpy
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import dump

leaveout_type = ["three_forecasting", "three_interpolation", "remove_recovery","all_times"][3]
test_tpsS_dict = {"three_forecasting":[16,17,18], "three_interpolation":[5,10,15], "remove_recovery":[5,7,9,11,15,16,17,18], "all_times":[]}
test_tps = test_tpsS_dict[leaveout_type]
n_tps=19
all_tps = list(range(n_tps)) 
train_tps = list(sorted(set(all_tps)-set(test_tps))) 

cnt_data = pd.read_csv("{}-norm_data-hvg.csv".format(leaveout_type), header=0, index_col=0)
meta_data = pd.read_csv("{}-meta_data.csv".format(leaveout_type),header=0, index_col=0)
cell_idx = np.where(~np.isnan(meta_data["day"].values))[0] # remove cells with nan labels
cnt_data = cnt_data.iloc[cell_idx, :]
meta_data = meta_data.loc[cnt_data.index,:]
cell_stage = meta_data["day"]
unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
cell_tp = np.zeros((len(cell_stage), ))
cell_tp[cell_tp == 0] = np.nan
for idx, s in enumerate(unique_cell_stages):
    cell_tp[np.where(cell_stage == s)[0]] = idx
meta_data["tp"] = cell_tp
ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)

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
