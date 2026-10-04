import torch
import torch.distributions as dist
import numpy as np
def NoisyMax(list,sigma,C,n):

    
    min_loss = min(list)
    
    min_index = list.index(min_loss)
    print(f'min_loss:{min_loss},min_index:{min_index}')

    
    laplace_noise = torch.tensor(np.random.exponential(C*sigma, size=len(list)), dtype=torch.float32, device='cpu')


    
    noised_list =  torch.tensor(list) + laplace_noise/n

    
    min_value, min_index = torch.min(noised_list, dim=0)

    print(f'min_value:{min_value},min_index:{min_index}')

    return min_index

