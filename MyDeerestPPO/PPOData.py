import numpy as np
import torch
from typing import Optional

class PPOData:
    def __init__(
            self,
            n_envs: int,
            steps: int,
            shape: Optional[list[int]] = None,
            dtype = torch.float,
            requires_grad = False,
            device = 'cpu'
        ):
        if type(shape) == type(None):
            self.shape = [steps, n_envs]
        else:
            self.shape = [steps, n_envs] + shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.reset()

    
    def reset(self):
        self.data = torch.zeros(
            self.shape,
            dtype = self.dtype,
            requires_grad=self.requires_grad
        ).to(self.device)


    def update(self, data, step: int, env : Optional[int] = None):
        if type(env) == int:
            self.data[step, env] = data
        else:
            self.data[step] = data


    def get_minibatch(self, indices: np.ndarray):
        return self.data[indices]

    
    def get(self, step: int):
        return self.data[step]
    
