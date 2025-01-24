# from Dev import model, train, utils, config
from Dev.train import run
from Dev.evaluation import  evaluate
from Dev.prediction import predict, predict_Endes, predict_TrajectoryPoints
from Dev.utils import init_all


name = 'Dev'
version = '1.0'

def train(args,leaveouts = None, holdout = None):
    if holdout is not None:
        # config = run_useEval(args,leaveouts=leaveouts, holdout=holdout)
        config = run(args,leaveouts=leaveouts, holdout=holdout)
    else:
        config = run(args,leaveouts=leaveouts, holdout=holdout)
    return config


def eval(args, use_loss='emd'):

    args.sde_adjoint = True
    args = init_all(args)
    args.for_train = False

    evaluate(args, use_loss=use_loss)
       

def derive_PredictionForTest(args, ts, n_cells=2000, ckpt_name='final'):
    data_listAllT, latent_xs_predict, recon_xs_predict = predict(args, ts, n_cells=n_cells, ckpt_name=ckpt_name)
    return data_listAllT, latent_xs_predict, recon_xs_predict

def derive_VectorEndes(args, ckpt_name='final', onlyUseDrift = True):
    data_listAllT, latent_listAllT, Endes_latent_allT = predict_Endes(args, ckpt_name=ckpt_name, onlyUseDrift = onlyUseDrift)
    return data_listAllT, latent_listAllT, Endes_latent_allT

def derive_trajectoryPoints(args, n_cells=2000, ckpt_name='final', dt = 0.1):
    data_listAllT, latent_xs_predict, recon_xs_predict, times = predict_TrajectoryPoints(args, n_cells=n_cells, ckpt_name=ckpt_name, dt = dt)
    return data_listAllT, latent_xs_predict, recon_xs_predict, times





