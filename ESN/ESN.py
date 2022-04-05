from re import L
from turtle import forward
from pygame import init
import torch
import torch.nn as tnn
from scipy import sparse
import numpy as np

class ESN():
    def __init__(self, res_units=100, out_units=1, connect_ratio=.2, spectral_radius=.99, readout_units=None, **kwargs) -> None:
        self.res_units = res_units
        self.out_units = out_units
        self.connect_ratio = connect_ratio
        self.spectral_radius = spectral_radius
        self.readout_units = readout_units
        # input scalling factor
        self.gamma = kwargs.get('gamma', 0.01)
        self.act_func = kwargs.get('act_func', torch.sigmoid)
        self.distr = kwargs.get('Wout_distr', 'uniform')
        # sigma of the output weights distribution if Gaussian 
        # or the absolute limit point for Uniform
        self.sigma = kwargs.get('sigma', 1.0)
        self.lr = kwargs.get('lr', 0.001)
        

        # Input weights depend on input size, they are set when data is provided
        self.Win = None
        # Reservoir weights
        self.W = self.initalise_weights()
        self.W.requires_grad = False
        # Readout if selected 
        if readout_units:
            self.Wout = self.init_output_weights
            self.Wout = torch.nn.Parameter(self.Wout, requires_grad = True)
            self.optimiser = kwargs.get('optimiser', torch.optim.Adam)(self.Wout, lr=self.lr)
            self.loss = kwargs.get('loss', torch.nn.MSELoss())
    
    def initalise_weights(self):
        #random unifom generator in the range of [-.5, .5]
        gen = lambda size: np.random.uniform(-.5, .5, size)
        #generate the sparse matrix
        W = sparse.random(self.res_units, self.res_units, density=self.connect_ratio, data_rvs=gen).todense()

        # Find eigenvalues
        E, _ = np.linalg.eig(W)
        e_max = np.max(np.abs(E))
        # Adjust spectral radius
        W /= e_max / self.spectral_radius
        return torch.from_numpy(W)
    
    def init_input_weights(self, X):
        #X shape of 
        # dim0-> samples/observations
        # dim1-> timesteps
        # dim2-> variables
        # N -> Obesrations, T-> timesteps, V-> variables
        N, T, V = X.shape
        if not self.Win:
            #TODO: there may be a better way to do this
            Win = self.gamma * np.random.randn(V, self.res_units)
            self.Win = torch.from_numpy(Win)
            self.Win.requires_grad = False
    
    def init_output_weights(self):
        if self.distr == 'uniform':
            wout = np.random.uniform(-self.sigma, self.sigma, (self.res_units, self.out_units)) / self.res_units
        elif self.distr == 'normal' or  self.distr == 'gaussian':
            wout = np.random.normal(0, self.sigma, (self.res_units, self.out_units)) / self.res_units

    def forward(self, X, Y):
        echoes = self.get_states()
        out = torch.mm(echoes, self.Wout)
        return out
    
    def fit(self, X, Y):
        for x, y in zip(X, Y):
            self.optimiser.zero_grad()
            out = self.forward(x)
            batch_loss = self.loss(out, y)
            batch_loss.backward()
            self.optimiser.step()
        pass

    def get_states(self, X):
        #X shape of 
        # dim0-> timesteps
        # dim1-> variables
        if not self.Win:
            self.Win = self.init_input_weights(X)
        
        h = torch.mm(X, self.Win)
        echoes = self.act_func(torch.mm(h, self.W))
        return echoes