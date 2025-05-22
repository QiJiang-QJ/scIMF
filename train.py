import torch
import torch.nn as nn
from torch import optim
import tqdm
import numpy as np
import os
from Dev.config import init_config
from Dev.utils import init_all, sampleGaussian
from Dev.load_Data import load_data
from Dev.model import MultiCNet
from time import strftime, localtime
import geomloss
import itertools





def run(args,leaveouts=None):

    args = init_all(args)

    if leaveouts is not None:
        args.task = 'leaveout'
        args.leaveouts=leaveouts
    else:
        args.task = 'fate'

    config = init_config(args)  
    config.for_train = True
    data_listAllT, _, config = load_data(config)
    data_listTrainT = [data_listAllT[t] for t in [0]+config.train_t]

    if os.path.exists(os.path.join(config.out_dir, 'done.log')):
        print(os.path.join(config.out_dir, 'done.log'), ' exists. Skipping.')
        
    else:
        model = MultiCNet(config)
        print(model)
        model.to(config.device)
        model.zero_grad()
        model.train()

        ot_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=config.sinkhorn_blur, 
                                         scaling=config.sinkhorn_scaling)
        torch.save(config.__dict__, config.config_pt)


        optimizer = optim.Adam(list(model.parameters()), lr=config.train_lr) # betas=(0.95, 0.99)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        pbar = tqdm.tqdm(range(config.train_epochs))

        best_train_loss_TrainT = np.inf
        with open(config.train_log, 'w') as log_handle:
            for epoch in pbar:

                losses_listTrainT = []
                losses_energy = []
                config.train_epoch = epoch
                optimizer.zero_grad()

                ts = [0] + args.train_t
                # ts = [np.float64(t) for t in ts]
                x0 = data_listAllT[0].to(config.device)
                latent_xs_energy_predict = model(ts, x0, batch_size=args.train_batch)
                
                num_TrainTs = len(config.train_t)
                for jj,train_t in enumerate(config.train_t):
                    loss_trainT = ot_loss(data_listAllT[int(train_t)].to(config.device), latent_xs_energy_predict[jj+1][:,0:-1])
                    losses_listTrainT.append(loss_trainT.item())

                    if (train_t == config.train_t[-1]) & (config.use_intLoss):
                        loss_energy = (torch.mean(latent_xs_energy_predict[-1][:,-1])) / train_t
                        losses_energy.append(loss_energy.item())
                        loss_all = ((loss_trainT * config.lambda_marginal)/num_TrainTs) + loss_energy
                    else:
                        loss_all = ((loss_trainT * config.lambda_marginal)/num_TrainTs)

                    loss_all.backward(retain_graph=True)

                train_loss_TrainT = np.mean(losses_listTrainT)
                if config.use_intLoss:
                    train_loss_energy = np.mean(losses_energy)

                if config.train_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train_clip)
                optimizer.step()
                scheduler.step()

                desc = "[train] {}".format(epoch + 1)
                desc += " {:.6f}".format(train_loss_TrainT)
                if config.use_intLoss:
                    desc += " {:.6f}".format(train_loss_energy)
                desc += " {:.6f}".format(best_train_loss_TrainT)
                pbar.set_description(desc)
                log_handle.write(desc + '\n')
                log_handle.flush()

                if train_loss_TrainT < best_train_loss_TrainT:
                    best_train_loss_TrainT = train_loss_TrainT
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': config.train_epoch + 1,
                    }, config.train_pt.format('best'))

                if (config.train_epoch + 1) % config.save == 0:
                    epoch_ = str(config.train_epoch + 1).rjust(6, '0')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': config.train_epoch + 1,
                    }, config.train_pt.format('epoch_{}'.format(epoch_)))

        config.done_log = os.path.join(config.out_dir, 'done.log')
        log_handle = open(config.done_log, 'w')
        timestamp = strftime("%a, %d %b %Y %H:%M:%S", localtime())
        log_handle.write(timestamp + '\n')
        log_handle.close()

    return config














