import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
import math
from torch import einsum
from einops import rearrange




#############################################################
###################### LAMBDA LAYER #########################
#############################################################
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

# lambda layer
expansion = 4

class LambdaLayer(nn.Module):
    def __init__(
        self,
        dim,
        dim_k=16,
        n = None,
        r = 23,
        heads = 4,
        dim_out = None,
        dim_u = 1):
        """
        params: 
            dim:    [int] in channels
            dim_out: [int] out channels 
            n           [int] size of the receptive window - max (height, width)
            r           [int] the receptive field for relative positional encoding (r x r)
            dim_k       [int] key dimension
            heads       [int] number of heads, for multi query
            dim_u       [int] 'intra-depth' dimension
        """
    
        super().__init__()
        dim_out = default(dim_out, dim)
        self.u = dim_u # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
        else:
            assert exists(n), 'You must specify the window size (n=h=w)'
            rel_lengths = 2 * n - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
            self.rel_pos = calc_rel_pos(n)

    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        k = k.softmax(dim=-1)

        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        if self.local_contexts:
            v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            λp = self.pos_conv(v)
            Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
        else:
            n, m = self.rel_pos.unbind(dim = -1)
            rel_pos_emb = self.rel_pos_emb[n, m]
            λp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b h v n', q, λp)

        Y = Yc + Yp
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return out



class LambdaBlock(nn.Module):
    expansion = expansion

    def __init__(self, in_planes, planes, stride=1, heads=4, dim_k=16, dim_u=1, r=23, n=None):
        """
        params:
            in_planes:  [int] in_channels
            planes:     [int] interplanes channels => out_channels = planes * expansion
            stride:     [int]
            n           [int] size of the receptive window - max (height, width). If r != None, then n is not used
            r           [int] the receptive field for relative positional encoding (r x r)
            dim_k       [int] key dimension
            heads       [int] number of heads, for multi query
            dim_u       [int] 'intra-depth' dimension.  [Default=1]
        """
        super(LambdaBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ModuleList(
            [
                LambdaLayer(dim=planes, dim_out=planes, 
                            dim_k=dim_k, dim_u=dim_u,
                            r=r, n=n, heads=heads
                            )
            ]
        )
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv2.append(nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1)))
        self.conv2.append(nn.BatchNorm2d(planes))
        self.conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*self.conv2)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LambdaStack(nn.Module):
    def __init__(self, in_planes, planes, downsample=False, num_block=1, heads=4, dim_k=16, dim_u=1, r=9, n=None):
        """
        params:
            in_planes:  [int] in_channels
            planes:     [int] interplanes channels => out_channels = planes * expansion
            downsample: [bool]  True is downsample feature maps
            num_block:  [int] The number of Lambda layers   
            n           [int] size of the receptive window - max (height, width). If r != None, then n is not used
            r           [int] the receptive field for relative positional encoding (r x r)
            dim_k       [int] key dimension
            heads       [int] number of heads, for multi query
            dim_u       [int] 'intra-depth' dimension.  [Default=1]
        """
        super(LambdaStack, self).__init__()
        layers = []
        for i in range(num_block):
            is_first = i == 0
            stride = 2 if is_first and downsample else 1
            layers.append(
                LambdaBlock(in_planes, planes, stride=stride, heads=heads, dim_k=dim_k, dim_u=dim_u, r=r, n=n)
            )
            in_planes = planes * LambdaBlock.expansion
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)




if __name__ == '__main__':
    # Modify num_block of LambdaStack or BottleStack to set the number of block
    inp = torch.randn(2,1024,11,11)
    lambdastack = LambdaStack(in_planes=1024, planes=256//4)
    lambda_out = lambdastack(inp)   
    print(lambda_out.shape)     # [1, 256, 11, 11]