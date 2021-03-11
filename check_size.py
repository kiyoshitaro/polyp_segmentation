from torchsummary import summary
import network.models as models

model = models.__dict__["GCPAGALDNetv8"]()
f = open("GCPAGALDNetv9.txt", "w")
summary(model, (3, 512, 512))