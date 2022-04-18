from echostate import ESN
import torch

esn = ESN.ESN(readout_units=1)
signal = torch.rand(10, 5)