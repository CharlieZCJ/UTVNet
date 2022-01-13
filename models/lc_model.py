import torch.nn as nn
from models import basicblock as B

class LIRCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=16):
        super(LIRCNN, self).__init__()
        self.model = B.IRCNN(in_nc, out_nc, nc)

    def forward(self, x):
        n = self.model(x)
        return x*n