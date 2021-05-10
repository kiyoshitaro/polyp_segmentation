import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
import math
from torch import einsum
from einops import rearrange


#############################################################
######################## MHSA BoT ###########################
#############################################################


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {"device": device, "dtype": dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim=3)
    flat_x = rearrange(x, "b h l c -> b h (l c)")
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1) :]
    return final_x


def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum("b h x y d, r d -> b h x y r", q, rel_k)
    logits = rearrange(logits, "b h x y r -> b (h x) y r")
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim=3, k=h)
    return logits


# positional embeddings


class AbsPosEmb(nn.Module):
    def __init__(self, fmap_size, dim_head):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, "h d -> h () d") + rearrange(
            self.width, "w d -> () w d"
        )
        emb = rearrange(emb, " h w d -> (h w) d")
        logits = einsum("b h i d, j d -> b h i j", q, emb)
        return logits


class RelPosEmb(nn.Module):
    def __init__(self, fmap_size, dim_head):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = self.fmap_size

        q = rearrange(q, "b h (x y) d -> b h x y d", x=h, y=w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b h x i y j-> b h (x y) (i j)")

        q = rearrange(q, "b h x y d -> b h y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b h x i y j -> b h (y x) (j i)")
        return rel_logits_w + rel_logits_h


# classes


class Attention(nn.Module):
    def __init__(self, *, dim, fmap_size, heads=4, dim_head=128, rel_pos_emb=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=heads), (q, k, v)
        )

        q *= self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        sim += self.pos_emb(q)

        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return out


class BottleBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out,
        proj_factor,
        downsample,
        heads=4,
        dim_head=128,
        rel_pos_emb=False,
        activation=nn.ReLU(),
    ):

        super().__init__()

        # shortcut

        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    dim,
                    dim_out,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(dim_out),
                activation,
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion

        attn_dim_in = dim_out // proj_factor
        attn_dim_out = heads * dim_head

        self.net = nn.Sequential(
            nn.Conv2d(dim, attn_dim_in, 1, bias=False),
            nn.BatchNorm2d(attn_dim_in),
            activation,
            Attention(
                dim=attn_dim_in,
                fmap_size=fmap_size,
                heads=heads,
                dim_head=dim_head,
                rel_pos_emb=rel_pos_emb,
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
        )

        # init last batch norm gamma to zero

        nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)


class BottleStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out=2048,
        proj_factor=4,
        num_block=1,
        heads=4,
        dim_head=128,
        downsample=False,
        rel_pos_emb=True,
        activation=nn.ReLU(),
    ):
        """
        dim             [int]   # channels in
        fmap_size       [int]   # feature map size
        dim_out         [int]   # channels out
        proj_factor     [int]   # projection factor
        num_block       [int]   # number of Bottleneck blocks
        downsample      [bool]  # downsample on first layer or not
        heads           [int]   # number of heads
        dim_head        [int]   # dimension per head, defaults to 128
        rel_pos_emb     [bool]  # use relative positional embedding - uses absolute if False
        activation      []      # activation throughout the network
        """
        super().__init__()
        # print(type(fmap_size))
        fmap_size = pair(fmap_size)
        # print(type(fmap_size))
        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_block):
            is_first = i == 0
            dim = dim if is_first else dim_out
            layer_downsample = is_first and downsample

            fmap_divisor = 2 if downsample and not is_first else 1
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(
                BottleBlock(
                    dim=dim,
                    fmap_size=layer_fmap_size,
                    dim_out=dim_out,
                    proj_factor=proj_factor,
                    heads=heads,
                    dim_head=dim_head,
                    downsample=layer_downsample,
                    rel_pos_emb=rel_pos_emb,
                    activation=activation,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert (
            c == self.dim
        ), f"channels of feature map {c} must match channels given at init {self.dim}"
        assert (
            h == self.fmap_size[0] and w == self.fmap_size[1]
        ), f"height and width ({h} {w}) of feature map must match the fmap_size given at init {self.fmap_size}"
        return self.net(x)


if __name__ == "__main__":
    # Modify num_block of LambdaStack or BottleStack to set the number of block
    inp = torch.randn(2, 1024, 11, 11)

    botstack = BottleStack(
        dim=1024, fmap_size=(11, 11), dim_out=256
    )  # Need to set feature map in BottleStack
    bot_out = botstack(inp)
    print(bot_out.shape)  # [1, 256, 11, 11]
