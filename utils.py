import torch
import numpy as np
import os
import random
import ot as pot
import scipy.sparse
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import torch.distributions as dist
from geomloss import SamplesLoss
from functools import partial
import math



def p_samp(p, num_sample, return_w = False):
    repflag = p.shape[0] < num_sample
    p_sub = np.random.choice(p.shape[0], size=num_sample, replace=repflag)
    w_ = torch.ones(len(p_sub))
    w_ = w_ / w_.sum()

    if isinstance(p, pd.DataFrame):
        p = torch.tensor(p.values, dtype=torch.float32)
    if return_w:
        return p[p_sub, :].clone(), w_
    else:
        return p[p_sub, :].clone()
    
def sampleGaussian(mean, std):

    d = dist.normal.Normal(torch.Tensor([0.]), torch.Tensor([1.]))
    r = d.sample(mean.size()).squeeze(-1).to(mean.device)
    x = r * std.float() + mean.float()
    return x
    

def init_device(arg):
    if torch.cuda.is_available() and arg.cuda is True:
        device = torch.device('cuda:{}'.format(arg.device_num))
    else:
        device = torch.device('cpu')
    print('using device: {}'.format(device))
    return device

def init_seed(arg):
    os.environ['PYTHONHASHSEED'] = str(arg.seed)
    random.seed(arg.seed)
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed(arg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def init_all(args):
    args.device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args

def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    


def _ot(true_data, pred_data, blur, scaling, device="cuda"):

    ot_solver = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, debias=True, backend="tensorized")
    if isinstance(true_data, np.ndarray):
        true_data = torch.DoubleTensor(true_data).to(device)
    elif isinstance(true_data, torch.Tensor):
        true_data = true_data.to(torch.double).to(device)

    if isinstance(pred_data, np.ndarray):
        pred_data = torch.DoubleTensor(pred_data).to(device)
    elif isinstance(pred_data, torch.Tensor):
        pred_data = pred_data.to(torch.double).to(device)
    ot_loss = ot_solver(true_data, pred_data).item()
    return ot_loss



