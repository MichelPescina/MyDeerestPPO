import time
import torch
import torch.nn as nn
from Models.Blocks import layer_init, DWSConv2d, SqueezeExcitation

class BasicBB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            DWSConv2d(3, 16, 5, 2, nn.Hardswish(), lambda l: layer_init(l)),
            SqueezeExcitation(16, 16*2//3, excitation_activ=nn.Hardsigmoid()),
            DWSConv2d(16, 32, 3, 2, nn.ReLU(), lambda l: layer_init(l)),
            SqueezeExcitation(32, 32//2, excitation_activ=nn.Hardsigmoid()),
            DWSConv2d(32, 64, 3, 2, nn.ReLU(), lambda l: layer_init(l)),
            SqueezeExcitation(64, 64//2, excitation_activ=nn.Hardsigmoid()),
            DWSConv2d(64, 96, 3, 2, nn.ReLU(), lambda l: layer_init(l)),
            SqueezeExcitation(96, 96//2, excitation_activ=nn.Hardsigmoid()),
            DWSConv2d(96, 128, 3, 2, nn.Hardswish(), lambda l: layer_init(l)),
            SqueezeExcitation(128, 128//2, excitation_activ=nn.Hardsigmoid()),
            DWSConv2d(128, 128, 5, 1, nn.Hardswish(), lambda l: layer_init(l)),
            SqueezeExcitation(128, 128//2, excitation_activ=nn.Hardsigmoid()),
            DWSConv2d(128, 128, 5, 1, nn.Hardswish(), lambda l: layer_init(l)),
            SqueezeExcitation(128, 128//2, excitation_activ=nn.Hardsigmoid()),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    import time
    from Blocks import count_parameters
    import numpy as np
    test = BasicBB(5)
    n = 100
    times = np.zeros(n)
    for i in range(n):
        input = torch.randn((1, 3, 240, 320))
        t1 = time.time()
        output = test(input)
        t2 = time.time()
        times[i] = t2 - t1
    print(f"Average time: {times.mean()}")
    print(f"Parameters: {count_parameters(test)}")
    print(output[0].shape)
