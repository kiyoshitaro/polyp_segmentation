from torchsummary import summary

import network.models as models

model = models.__dict__["GCPAPSPSmallNet"]()
# f = open("GCPAGALDNetv9.txt", "w")
summary(model, (3, 352, 352))