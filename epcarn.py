import torch
import torch.nn as nn
import ops
from rcarn_block import Net as RCARN

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock, nFrames, scale_factor):
        super(Net, self).__init__()
        self.base_channels = 32  #64

        self.mid = nn.Conv2d(3, self.base_channels, 3, 1, 1)
        self.rcarn = RCARN()
        self.mid2 = nn.Conv2d(32, self.base_channels, 3, 1, 1)

        self.share = ops.ResidualBlock(self.base_channels, self.base_channels)
        self.shareConv = nn.Conv2d(self.base_channels, self.base_channels, 3, 1, 1)

        self.upsample = ops.UpsampleBlock(self.base_channels, scale=2, 
                                          multi_scale=False,
                                          group=1)

        self.output = nn.Conv2d(self.base_channels, 3, 3, 1, 1)

    def forward(self, x, ref):

        #feat_input = self.entry(x)
        feat_input = x

        out = self.rcarn(feat_input, ref, scale = 2) #First Model, Third Mode

        ####Reconstruction
            
        output = self.mid2(out)
        
        output = self.upsample(output, 2)
        #print(output.size())  
        output = self.output(output)

        refing = "Loop"
        return output
