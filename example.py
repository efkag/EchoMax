from echostate import ESN
import torch

esn = ESN.ESN(readout_units=1)
signal = torch.rand(10, 5, dtype=torch.float64)
print(signal.dtype)

states =  esn.get_states(signal)
print(states.size())

out =  esn.forward(signal)
print(out.size())