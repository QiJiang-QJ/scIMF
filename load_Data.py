import torch
import numpy as np
import os
import pandas as pd
import scanpy
import natsort


def load_data_ZB(args):
    if args.leaveouts == [4,6,8]:
        leaveout_type = 'three_interpolation'
    elif args.leaveouts == [10,11]:
        leaveout_type = 'two_forecasting'
    elif args.leaveouts == [2,4,6,8,10,11]:
        leaveout_type = 'remove_recovery'
    elif args.leaveouts == []:
        leaveout_type = 'all_times'
    args.split_type = leaveout_type
    data = pd.read_csv("{}/processed/{}-{}_data.csv".format(args.data_dir,leaveout_type, args.latent_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/processed/meta_data.csv".format(args.data_dir),header=0, index_col=0)
    meta_data = meta_data.loc[data.index,:]
    cell_stage = meta_data["stage.nice"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_types =  meta_data['celltype'].values
    return data.to_numpy(), cell_tp, cell_types



def load_data_MEF(args):
    if args.leaveouts == [5,10,15]:
        leaveout_type = 'three_interpolation'
    elif args.leaveouts == [16,17,18]:
        leaveout_type = 'three_forecasting'
    elif args.leaveouts == [5,7,9,11,15,16,17,18]:
        leaveout_type = 'remove_recovery'
    elif args.leaveouts == []:
        leaveout_type = 'all_times'
    args.split_type = leaveout_type
    data = pd.read_csv("{}/reduce_processed/{}-{}_data.csv".format(args.data_dir, leaveout_type, args.latent_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/reduce_processed/{}-meta_data.csv".format(args.data_dir, leaveout_type), header=0, index_col=0)
    cell_idx = np.where(~np.isnan(meta_data["day"].values))[0] # remove cells with nan labels
    data = data.iloc[cell_idx, :]
    meta_data = meta_data.loc[data.index,:]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_types =  None
    return data.to_numpy(), cell_tp, cell_types

def load_data_Panc(args):
    if args.leaveouts == []:
        leaveout_type = 'all_times'
    elif args.leaveouts == [1]:
        leaveout_type = 'leaveout1'
    elif args.leaveouts == [2]:
        leaveout_type = 'leaveout2'
    elif args.leaveouts == [3]:
        leaveout_type = 'leaveout3'
    elif args.leaveouts == [4]:
        leaveout_type = 'leaveout4'
    elif args.leaveouts == [5]:
        leaveout_type = 'leaveout5'
    elif args.leaveouts == [6]:
        leaveout_type = 'leaveout6'
    elif args.leaveouts == [7]:
        leaveout_type = 'leaveout7'
    args.split_type = leaveout_type
    data = pd.read_csv("{}/new_processed/{}-{}_data.csv".format(args.data_dir,leaveout_type, args.latent_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/new_processed/meta_data.csv".format(args.data_dir),header=0, index_col=0)
    meta_data = meta_data.loc[data.index,:]
    cell_stage = meta_data["CellWeek"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_types =  meta_data['Assigned_cluster'].values
    return data.to_numpy(), cell_tp, cell_types




def load_data(args):

        if args.for_train == False:
        args.out_dir = "/".join(args.out_dir.split("/")[:2])
        args.data_dir = "/".join(args.data_dir.split("/")[:2])

    args.data_dir = "{}/{}".format(args.data_dir,args.dataset)

    if args.dataset == 'ZB':
        n_tps = 12
        args.Train_ts = list(range(1, n_tps)) # 1-11
        args.train_t = list(sorted(set(args.Train_ts)-set(args.leaveouts)))   
        args.test_t = args.leaveouts
        data, cell_tps, cell_types = load_data_ZB(args)    
    elif args.dataset == 'MEF':
        n_tps = 19
        args.Train_ts = list(range(1, n_tps)) # 1-18
        args.train_t = list(sorted(set(args.Train_ts)-set(args.leaveouts))) 
        args.test_t = args.leaveouts
        data, cell_tps, cell_types = load_data_MEF(args)  
    elif args.dataset == 'Panc':
        n_tps = 8
        args.Train_ts = list(range(1, n_tps)) # 1-7
        args.train_t = list(sorted(set(args.Train_ts)-set(args.leaveouts))) 
        args.test_t = args.leaveouts
        data, cell_tps, cell_types = load_data_Panc(args)

    args.out_dir = "{}/{}/{}/seed_{}/Ours".format(args.out_dir, args.dataset, args.split_type, args.seed)

    print('--------------------------------------------')
    print('----------leaveout_t=',args.leaveouts,'---------')
    print('----------train_t=', args.train_t)
    print('--------------------------------------------')

     
    data_listAllT = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(0, n_tps)]  # (# tps, # cells, # genes)
    if cell_types is not None:
        cell_types_listAllT = [cell_types[np.where(cell_tps == t)[0]] for t in range(0, n_tps)]
    else:
        cell_types_listAllT = None

    args = constructOutDir(args)

    return data_listAllT, cell_types_listAllT, args



def constructOutDir(args):
    if not os.path.exists(args.out_dir):
        print('Making directory at {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    else:
        print('Directory exists at {}'.format(args.out_dir))

    args.train_pt=os.path.join(args.out_dir, 'train.{}.pt')
    args.done_log=os.path.join(args.out_dir, 'done.log')
    args.config_pt=os.path.join(args.out_dir, 'config.pt')
    args.train_log = os.path.join(args.out_dir, 'train.log')
    
    return args


def return_SplitTypes(data_name):
    if data_name == 'zebrafish':
        split_types = ["two_forecasting", "three_interpolation", "remove_recovery"]
        test_tpsS = [[10,11], [4,6,8], [2,4,6,8,10,11]]
    elif data_name == 'drosophila':
        split_types = ["three_forecasting", "three_interpolation", "remove_recovery"]
        test_tpsS = [[8,9,10], [4,6,8], [2,4,6,8,9,10]]
    elif data_name == 'wot':
        split_types = ["three_forecasting", "three_interpolation", "remove_recovery"]
        test_tpsS = [[16,17,18], [5,10,15], [5,7,9,11,15,16,17,18]]
    elif data_name == 'Veres':
        split_types = ["two_forecasting", "two_interpolation"]
        test_tpsS = [[6,7], [3,5]]
    elif data_name == 'DIISCO_CARB':
        split_types = ["forecasting", "interpolation", "remove_recovery"]
        test_tpsS = [[4], [2], [2,4]]

    return split_types, test_tpsS
