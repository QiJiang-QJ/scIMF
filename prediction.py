import torch
import numpy as np
import pandas as pd
import os
import glob

from Dev.utils import init_all
from Dev.load_Data import load_data
from Dev.model import MultiCNet


## 已经导入train的config信息
def derive_model(args, ckpt_name='best'):

    print(args.train_pt)

    args = init_all(args)
    args.for_train = False
    args.sde_adjoint = True

    data_listAllT, cell_types_listAllT, config = load_data(args)
    model = MultiCNet(config)

    print(config.train_pt)

    if ckpt_name == 'final':
        epoch_ = str(config.train_epochs).rjust(6, '0')
        ckpt_name = 'epoch_{}'.format(epoch_)
    elif ckpt_name == 'best':
        ckpt_name = 'best'
    train_pt = "./" + config.train_pt.format(ckpt_name)
    print(config.train_pt)
    print(train_pt)
    checkpoint = torch.load(train_pt, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(config.device)
    model.eval()

    return model, data_listAllT, cell_types_listAllT, config






def predict(args, ts, n_cells=None, ckpt_name='best'):
    model, latent_listAllT, cell_types_listAllT, config = derive_model(args, ckpt_name=ckpt_name)

    if n_cells is None:
        n_cells = latent_listAllT[0].shape[0]
    latent_xs_predict = model.predict(ts, latent_listAllT[0], n_cells=n_cells)
    
    latent_xs_predict = [
        item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
        for item in latent_xs_predict
    ]
 
    return latent_listAllT, latent_xs_predict





def predict_Endes(args, ckpt_name='best'):

    model, latent_listAllT, cell_types_listAllT, config = derive_model(args, ckpt_name=ckpt_name)

    Endes_latent_allT = []
    for ii, latent_xt in enumerate(latent_listAllT):
        latent_xt = latent_xt.to(config.device)
        latent_xt_endes = latent_xt + model.drift(None,latent_xt)
        Endes_latent_allT.append(latent_xt_endes.detach().cpu().numpy())
        
    return latent_listAllT, Endes_latent_allT



def predict_TrajectoryPoints(args, n_cells=2000, ckpt_name='final', dt = 0.1):
    model, latent_listAllT, cell_types_listAllT, config = derive_model(args, ckpt_name=ckpt_name)

    times = np.round(np.arange(0, len(latent_listAllT)-1+dt, dt), 3).tolist()
    times = [np.float64(t) for t in times]

    latent_xs_predict = model.predict(times, latent_listAllT[0], n_cells=n_cells)
    
    latent_xs_predict = [
        item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
        for item in latent_xs_predict
    ]

    
    return latent_listAllT, latent_xs_predict, times





