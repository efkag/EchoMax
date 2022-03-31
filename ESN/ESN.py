from turtle import forward
from pygame import init
import torch
import torch.nn as tnn
from scipy import sparse
import numpy as np

class ESN():
    def __init__(self, res_units=100, connect_ratio=.2, spectral_radius=.99, readout_units=None) -> None:
        self.res_units = res_units
        self.connect_ratio = .2
        self.spectral_radius = spectral_radius
        self.readout_units = readout_units
        # Input weights depend on input size, they are set when data is provided
        self.Win = None
        # Reservoir weights
        self.W = self.initalise_weights() 
        # Readout if selected 
        if readout_units:
            self.Wout = None
        self.input_scaling = .2

    
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
        # N -> Obesrations, T-> timesteps, V-> variables
        N, T, V = X.shape
        if not self.Win:
            #TODO: there may be a better way to do this
            Win = (2.0*np.random.binomial(1, 0.5 , [self.res_units, V]) - 1.0)*self._input_scaling
            self.Win = torch.from_numpy(Win)
        pass

    def forward(sefl, X):
        pass
    
    def fit(self):
        pass

    def get_states(self, X):
        if not self.Win:
            self.Win = self.init_input_weights(X)
        
        
        pass