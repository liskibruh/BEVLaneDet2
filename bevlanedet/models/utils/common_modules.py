from torch import nn

def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, 1)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod

class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super().__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.module(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        
        return self.relu(out)