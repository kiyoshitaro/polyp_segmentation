import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# import sys

# sys.path.append("/home/admin_mcn/hung/polyp_segmentation")


class SizeEstimator(object):
    def __init__(self, model, input_size=(1, 1, 32, 32), bits=32):
        """
        Estimates the size of PyTorch models in memory
        for a given input size
        """
        self.model = model
        self.input_size = input_size
        self.bits = bits
        return

    def get_parameter_sizes(self):
        """Get sizes of all parameters in `model`"""
        mods = list(self.model.modules())
        sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        """Run sample input through each layer to get output sizes"""
        input_ = Variable(torch.FloatTensor(*self.input_size), volatile=True)
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        """Calculate total number of bits to store `model` parameters"""
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        """Calculate bits to store forward and backward pass"""
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = total_bits * 2
        return

    def calc_input_bits(self):
        """Calculate bits to store input"""
        self.input_bits = np.prod(np.array(self.input_size)) * self.bits
        return

    def estimate_size(self):
        """Estimate model size in memory in megabytes and bits"""
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total / 8) / (1024 ** 2)
        return total_megabytes, total


if __name__ == "__main__":
    # import network.models as models
    # from utils.pytorch_calsize import SizeEstimator

    # model = models.__dict__["GCPAGALDNetv7"]()
    # se = SizeEstimator(model, input_size=(1, 3, 352, 352))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchsummary import summary

    class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()

            self.conv0 = nn.Conv2d(1, 16, kernel_size=3, padding=5)
            self.conv1 = nn.Conv2d(16, 32, kernel_size=3)

        def forward(self, x):
            h = self.conv0(x)
            h = self.conv1(h)
            return h
    model = Net()
    device = torch.device("cpu") # PyTorch v0.4.0
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = Net().to(device)


    import network.models as models
    # model = models.__dict__["PraNetvGALD"]()
    model = models.__dict__["GCPAGALDNetv4"]()
    # HardnetMSEG
    # PraNetvGALD
    from network.encoders import hardnet
    model = hardnet()
    # from network.encoders import res2net50_v1b_26w_4s
    # model = res2net50_v1b_26w_4s()
    
    # model.cuda()
    summary(model, (3, 352, 352))
