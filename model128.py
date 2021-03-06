import torch
import torch.nn as nn
import math
class block(nn.Module):
    def __init__(self, n, nout=None):
        super().__init__()
        if not nout:
            nout = n
        self.conv = nn.Conv2d(n, nout, (3,3), padding=1)#, bias=False)
    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.elu(x)
        return x

class G(nn.Module):
    def __init__(self, n=128, h=64):
        super().__init__()
        self.decoder = Decoder(n=n, h=h)

    def forward(self, x):
        x = self.decoder(x)#.clamp(min=-10, max=10)
        return x

class D(nn.Module):
    def __init__(self, n=128, h = 64):
        super().__init__()
        self.encoder = Encoder(n=n, h=h)
        self.decoder = Decoder(n=n, h=h)
    def forward(self, x):
        x = self.encoder(x)
        z = x.clone()#.detach()#nn.functional.tanh(x)
        x = self.decoder(z)
        return x, z#.detach()

class Encoder(nn.Module):
    def __init__(self, n=128, h=64):
        super().__init__()
        self.main = nn.Sequential(
        block(3,n),
        block(n),
        block(n),
        block(n, n*2),
        nn.MaxPool2d((2,2),2),
        block(n*2),
        block(n*2),
        block(n*2, n*3),
        nn.MaxPool2d((2,2),2),
        block(n*3),
        block(n*3),
        block(n*3, n*4),
        nn.MaxPool2d((2,2),2),
        block(n*4),
        block(n*4),
        block(n*4, n*5),
        nn.MaxPool2d((2,2),2),
        block(n*5),
        block(n*5),
        nn.Conv2d(n*5, h, (8,8)),
        )
    def forward(self, x):
        x = self.main(x)
        return x

class Decoder(nn.Module):
    def __init__(self, n=128, h=64):
        super().__init__()
        self.linear = nn.Linear(h, n*8*8)
        
        self.main = nn.Sequential(
        #nn.ConvTranspose2d(h, n, 4, 1, 0),
        block(n), 
        #block(n), 
        nn.UpsamplingNearest2d(scale_factor=2),
        block(n),
        #block(n),
        nn.UpsamplingNearest2d(scale_factor=2),
        block(n),
        #block(n),
        nn.UpsamplingNearest2d(scale_factor=2),
        block(n),        
        #block(n),        
        nn.UpsamplingNearest2d(scale_factor=2),
        block(n),        
        #block(n),  
        nn.Conv2d(n, 3, (3,3), padding=1, bias=False)
        )
    def forward(self, x):
        x = x.view(-1,64)        
        x = self.linear(x)
        x = x.view(-1, 128, 8,8)
        x = nn.functional.elu(x)
        x = self.main(x)
        return x

