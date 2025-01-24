import torch
import Dev.torchsde_addfloat as torchsde
import torch.nn as nn
from Dev.utils import p_samp
import torch.distributions as dist
from Dev.latentsde_layers import Drift_inter, Drift_intra, Diffusion




class MultiCNet(nn.Module):

    def __init__(self, args):
        super(MultiCNet, self).__init__()        
        
        self.latent_dim = args.latent_dim
        self.sde_adjoint = args.sde_adjoint
        self.dt = args.train_dt
        self.device = args.device

        if args.sde_adjoint:
            self.sde_type = 'ito'
            self.method = 'euler'
            self.adjoint_method = 'euler'
        else:
            self.sde_type = 'ito'
            self.method = 'euler'


        self.drift_inter = Drift_inter(args)
        self.drift_intra = Drift_intra(args)

        self.noise_type = 'diagonal'
        self.diffusion = Diffusion(args)
        
    def f(self, t, x_and_energy):
        x = x_and_energy[:,0:-1]
        drift = self.drift(t,x)
        drift_energy = (torch.norm(drift,dim=1)**2).unsqueeze(1)
        return torch.cat([drift, drift_energy], dim=1)
    
    def drift(self, t, x):
        return self.drift_inter(t,x) + self.drift_intra(t,x)
    
    def g(self, t, x_and_energy):
        x = x_and_energy[:,0:-1]
        extra_g = (((torch.zeros(x.shape[0]).to(x.device))).unsqueeze(1))
        return torch.cat([self.diffusion(t,x), extra_g], dim=1)

    def forward(self, t, x0, batch_size=None, return_whole_sequence=True):

        if batch_size is not None:
            x0 = p_samp(x0, batch_size)
        energy_0 = torch.zeros(x0.shape[0]).unsqueeze(1).to(x0.device)
        x_and_energy = torch.cat([x0,energy_0], dim=1)

        if self.sde_adjoint:
            x_and_energy = torchsde.sdeint_adjoint(self, x_and_energy, t, 
                                        method=self.method, adjoint_method=self.adjoint_method,
                                        dt=self.dt, dt_min=0.01)
        
        else:
            x_and_energy = torchsde.sdeint(self, x_and_energy, t, 
                                method=self.method, 
                                dt=self.dt)
    
        x_and_energy = x_and_energy if return_whole_sequence else x_and_energy[-1] 
        return x_and_energy
    
    def predict(self, t, data_t0, n_cells=None):
        data_t0 = data_t0.to(self.device)
        latent_xs_predict = self.forward( t, data_t0, batch_size=n_cells)[:,:,0:-1]
        return latent_xs_predict

        
    
    



   
    














