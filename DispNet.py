import torch.nn as nn
import torch

kn = torch.Tensor([[0.00001 ,0.00002, 0.00001],[0.00002, 0.00004 ,0.00002], [0.00001, 0.00002 ,0.00001]]).unsqueeze(0).unsqueeze(0)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, bias):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = kernel, stride = stride, padding = padding, bias = bias),
            nn.BatchNorm2d(num_features = out_ch),
            nn.LeakyReLU()
        )

    def forward(self, input):
        return self.conv(input)

class DispNet(nn.Module):
    def __init__(self):
        super(DispNet, self).__init__()

        self.conv1 = ConvBlock(6, 64, 7, stride = 2, padding = 3, bias = False)
        self.conv2 = ConvBlock(64, 128, 5, stride=2, padding=2, bias = False)
        self.conv3a = ConvBlock(128, 256, 5, stride=2, padding=2, bias = False)
        self.conv3b = ConvBlock(256, 256, 3, stride=1, padding=1, bias = False)
        self.conv4a = ConvBlock(256, 512, 3, stride=2, padding=1, bias = False)
        self.conv4b = ConvBlock(512, 512, 3, stride=1, padding=1, bias = False)
        self.conv5a = ConvBlock(512, 512, 3, stride=2, padding=1, bias = False)
        self.conv5b = ConvBlock(512, 512, 3, stride=1, padding=1, bias = False)
        self.conv6a = ConvBlock(512, 1024, 3, stride=2, padding=1, bias = False)
        self.conv6b = ConvBlock(1024, 1024, 3, stride=1, padding=1, bias = False)

        self.pr1 = nn.Conv2d(32, 1, 3, stride=1, padding=1, bias = False)
        self.pr1.weight = nn.Parameter(kn.expand(1, 32, -1, -1))

        self.upconv5 = ConvBlock(1024, 512, 3, stride=1, padding=1, bias = False)
        self.upconv4 = ConvBlock(512, 256, 3, stride=1, padding=1, bias = False)
        self.upconv3 = ConvBlock(256, 128, 3, stride=1, padding=1, bias = False)
        self.upconv2 = ConvBlock(128, 64, 3, stride=1, padding=1, bias = False)
        self.upconv1 = ConvBlock(64, 32, 3, stride=1, padding=1, bias = False)

        self.iconv5 = ConvBlock(1024, 512, 3, stride=1, padding=1, bias = False)
        self.iconv4 = ConvBlock(768, 256, 3, stride=1, padding=1, bias = False)
        self.iconv3 = ConvBlock(384, 128, 3, stride=1, padding=1, bias = False)
        self.iconv2 = ConvBlock(192, 64, 3, stride=1, padding=1, bias = False)
        self.iconv1 = ConvBlock(96, 32, 3, stride=1, padding=1, bias = False)

        self.up = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.loss = nn.SmoothL1Loss()







    def forward(self, x, y):

        total_loss = 0

        # ============== ENCODE ================
        conv1 = self.conv1(x.float())

        conv2 = self.conv2(conv1)

        conv3a = self.conv3a(conv2)

        conv3b = self.conv3b(conv3a)

        conv4a = self.conv4a(conv3b)

        conv4b = self.conv4b(conv4a)

        conv5a = self.conv5a(conv4b)

        conv5b = self.conv5b(conv5a)

        conv6a = self.conv6a(conv5b)

        conv6b = self.conv6b(conv6a)

        # =============== DECODE ==================
        upconv5 = self.upconv5(self.up(conv6b))
        
        concat = torch.cat((upconv5, conv5b), 1)  
        iconv5 = self.iconv5(concat)

        upconv4 = self.upconv4(self.up(iconv5))

        concat = torch.cat((upconv4, conv4b), 1)
        iconv4 = self.iconv4(concat)

        upconv3 = self.upconv3(self.up(iconv4))

        concat = torch.cat((upconv3, conv3b), 1)
        iconv3 = self.iconv3(concat)

        upconv2 = self.upconv2(self.up(iconv3))

        concat = torch.cat((upconv2, conv2), 1)
        iconv2 = self.iconv2(concat)

        upconv1 = self.upconv1(self.up(iconv2))

        concat = torch.cat((upconv1, conv1), 1)
        iconv1 = self.iconv1(concat)

        pr1 = self.pr1(iconv1)
        pr1 = pr1.squeeze(1)

        return pr1
