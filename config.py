import argparse
import os

def config():
    parser = argparse.ArgumentParser()
        # system config
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--device_num', default=1)
    parser.add_argument('--data_dir', default='../Data')
    parser.add_argument('--out_dir',default='./Output')

        # data config
    parser.add_argument('--dataset', type=str, default='Veres')
    parser.add_argument('--latent_dim', type=int, default=50)


        # model_Drift_inter config
    parser.add_argument('--inter_trm_num_head', type=int, default=2)
    parser.add_argument('--inter_trm_num_layers', type=int, default=1)
    parser.add_argument('--attn_drop_ratio', type=float, default=0.1)


        # model_Drift_intra config
    parser.add_argument('--intra_dims', default=[256,256])

    
        # model_Diffusion config
    parser.add_argument('--sigma_type', default='const')
    parser.add_argument('--sigma_const', type=float, default=0.1)
    parser.add_argument('--sde_adjoint', action='store_true', default=False)


        # training config
    parser.add_argument('--train_epochs', type=int, default=3000)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--train_batch', type=float, default=512)
    parser.add_argument('--train_dt', type=float, default=0.1)
    parser.add_argument('--train_clip',type=float, default=0.1)
    parser.add_argument('--save', type=int, default=500)
    parser.add_argument('--lambda_marginal', type=float, default=1)

        # loss config
    parser.add_argument('--sinkhorn_scaling',type=float, default=0.7)
    parser.add_argument('--sinkhorn_blur',type=float, default=0.1)
    parser.add_argument('--use_intLoss', default=True)

        # evaluation config
    parser.add_argument('--evaluate_n', type=int, default=2000)

    args = parser.parse_known_args()[0]
    return args



def init_config(args):
    
    args.model_setting = (
        "{train_lr}-"
        "{train_batch}-"
        "{lambda_marginal}-"
        "{inter_trm_num_layers}"
    ).format(**args.__dict__)

    return args
