import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_1(in_c, out_c, bt):
    if bt:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1 ,bias= False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1 ,bias= False),
            nn.LeakyReLU(0.2, inplace=True)
        )
    return conv

def de_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1 ,bias= False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )
    return conv

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = 1
        self.nef = 32
        self.ngf = 16
        self.nBottleneck = 32
        self.nc = 1
        self.num_same_conv = 5
        
        # 1x80x80
        self.conv1 = conv_1(self.nc, self.nef, False)
        # 32x40x40
        self.conv2 = conv_1(self.nef,self.nef, True)
        # 32x20x20
        self.conv3 = conv_1(self.nef, self.nef*2, True)
        # 64x10x10
        self.conv4 = conv_1(self.nef*2+1, self.nef*4, True)
        # 128x5x5
        self.conv6 = nn.Conv2d(self.nef*4,self.nBottleneck,2, bias=False)
        # 4000x4x4
        self.batchNorm1 = nn.BatchNorm2d(self.nBottleneck)
        self.leak_relu = nn.LeakyReLU(0.2, inplace=True)
        # 4000x4x4
        
        self.num_same_conv=self.num_same_conv
        self.sameconvs = nn.ModuleList([nn.ConvTranspose2d(32,32,3,1,1,bias=False) for _ in range(self.num_same_conv)])
        self.samepools = nn.ModuleList([nn.MaxPool2d(kernel_size=3,stride=1,padding=1) for _ in range(self.num_same_conv)])
        self.samebns = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(self.num_same_conv)])
        
        
        self.convt1 = nn.ConvTranspose2d(self.nBottleneck, self.ngf * 8, 2, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(self.ngf * 8)
        self.relu = nn.ReLU(True)
        # 128x5x5
        self.convt2 = de_conv(256, 64)
        # 64x10x10
        self.convt3 = de_conv(128+1, 32)
        # 32x20x20
        self.convt4 = de_conv(64, 32)
        # 32x40x40
        self.convt6 = nn.ConvTranspose2d(64, self.nc, 4, 2, 1, bias=False)
        # 1x80x80
        self.tan = nn.Tanh()

    def forward(self,noise,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        mod_input = torch.cat([noise,x3],dim=1)
        x4 = self.conv4(mod_input)
        x6 = self.conv6(x4)
        x7 = self.batchNorm1(x6)
        x8 = self.leak_relu(x7)        
        x9 = self.convt1(x8)
            
        x10 = self.batchNorm2(x9)
        x11 = self.relu(x10)
        x12 = self.convt2(torch.cat([x4, x11], 1))
        out = self.convt3(torch.cat([mod_input, x12], 1)) 
        
        for i in range(self.num_same_conv):
            conv = self.sameconvs[i]
            pool = self.samepools[i]
            bn = self.samebns[i]
            
            out = conv(out)
            out = pool(out)
            out = bn(out)
            out = F.leaky_relu(out,negative_slope=0.2)
            
        x14 = self.convt4(torch.cat([x2, out], 1))
        x15 = self.convt6(torch.cat([x1, x14], 1))
        
        return self.tan(x15)
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngpu = 1
        
        # input is (nc) x 80 x 80
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1, bias=False)
        
        # state size. (ndf) x 40 x 40
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(64, affine=True)
        
        # state size. (ndf*2) x 20 x 20
        self.conv3 = nn.Conv2d(64, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.InstanceNorm2d(64, affine=True)
        
        # state size. (ndf*4) x 10 x 10
#         self.conv4 = nn.Conv2d(64+1, 32, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(64, 32, 4, 2, 1, bias=False)
        self.bn4 = nn.InstanceNorm2d(32, affine=True)
        
        # state size. (ndf*8) x 4 x 4
        self.conv5=nn.Conv2d(32, 1, 5, 1, 0, bias=False)
#         self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.conv1(input)
        out = F.leaky_relu(out,negative_slope=0.2)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.leaky_relu(out,negative_slope=0.2)
    
        out = self.conv3(out)
    
        out = self.bn3(out)
        out = F.leaky_relu(out,negative_slope=0.2)
        
        out = torch.cat([out],dim=1)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out = F.leaky_relu(out,negative_slope=0.2)
        out = self.conv5(out)
        
        # removing sigmoid activation for wgan
#         out = self.sigmoid(out)
        
        return out