import torch
import torch.nn as nn
import torch.nn.functional as F
import ops

class Block(nn.Module):
    def __init__(self, 
                 in_channels = 64, out_channels = 64, base_channels = 32,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.EResidualBlock(base_channels, base_channels)
        self.b2 = ops.EResidualBlock(base_channels, base_channels)
        #self.b3 = ops.EResidualBlock(base_channels, base_channels)
        self.c1 = ops.BasicBlock(base_channels*2, base_channels, 1, 1, 0)
        self.c2 = ops.BasicBlock(base_channels*3, base_channels, 1, 1, 0)
        #self.c3 = ops.BasicBlock(base_channels*4, base_channels, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        '''
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        '''
        return o2
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.base_channels = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(1, self.base_channels, 3, 1, 1, bias=False).to(self.device)
        torch.nn.init.kaiming_normal_(self.conv1.weight)

        self.entry1 = nn.Conv2d(3, self.base_channels, 3, 1, 1, bias=False).to(self.device)
        torch.nn.init.kaiming_normal_(self.entry1.weight)
        self.entry2 = nn.Conv2d(32, self.base_channels, 3, 1, 1, bias=False).to(self.device)
        torch.nn.init.kaiming_normal_(self.entry2.weight)

        self.b1 = Block(base_channels=32).to(self.device)
        #self.b2 = Block(base_channels=32)
        #self.b3 = Block(base_channels=32)
        self.c1 = ops.BasicBlock(self.base_channels*2, self.base_channels, 1, 1, 0).to(self.device)
        #self.c2 = ops.BasicBlock(32*3, 64, 1, 1, 0)
        #self.c3 = ops.BasicBlock(32*4, 64, 1, 1, 0)
        self.down_feat = ops.BasicBlock(32*2, 32, 1, 1, 0).to(self.device)
        
        #self.upsample = ops.UpsampleBlock(64, scale=scale, multi_scale=multi_scale, group=group)
        #self.exit = nn.Conv2d(64, 3, 3, 1, 1)
                
    def forward(self, x, ref, scale):
        #prepare reference(edge_feature)
        ref = self.conv1(ref)
        ref = F.leaky_relu(ref,inplace=True)
        ref = F.max_pool2d(ref,2)

        x = self.entry1(x)

        x = F.leaky_relu(x,inplace=True)
        lr = torch.cat([x,ref],dim=1)
        
        #perform CARN
        lr = self.down_feat(lr)
        sr = self.carn(lr)

        return sr

    def carn(self, ip):
        b1 = self.b1(ip)
        c1 = torch.cat([ip, b1], dim=1)
        o1 = self.c1(c1)

        return o1

    def init_w(self,m):
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)