import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from utils.warplayer import forwardwarp
from model.convlstm import ConvLSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)
        # end
    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)
        ], 1
    )
    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True
    )


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.bernoulli(x)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PReLU(out_planes)
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std).to(device)
        self.weight.data = torch.eye(c).view(c, c, 1, 1).to(device)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean).to(device)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean).to(device)
        self.requires_grad = False


class Head(nn.Module):
    def __init__(self, c):
        super(Head, self).__init__()
        model = torchvision.models.resnet18(pretrained=False)
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)                        
        self.cnn0 = nn.Sequential(*nn.ModuleList(model.children())[: 3])
        self.cnn1 = nn.Sequential(
            *list(model.children())[3: 5],
        )
        self.cnn2 = nn.Sequential(
            *list(model.children())[5: 6],
        )
        self.out0 = nn.Conv2d(64, c, 1, 1, 0)
        self.out1 = nn.Conv2d(64, c, 1, 1, 0)
        self.out2 = nn.Conv2d(128, c, 1, 1, 0)
        self.upsample = deconv(c, c // 2)

    def forward(self, x):
        x = self.normalize(x)
        f0 = self.cnn0(x)
        f1 = self.cnn1(f0)
        f2 = self.cnn2(f1)
        f0 = self.out0(f0)
        f1 = F.interpolate(self.out1(f1), scale_factor=2.0, mode="bilinear")
        f2 = F.interpolate(self.out2(f2), scale_factor=4.0, mode="bilinear")
        return self.upsample(f0 + f1 + f2)


class MaskBlock(nn.Module):     # ResNet, only change c from in_dim to out_dim, h & w remain unchanged
    def __init__(self, in_dim, out_dim, depth=1):
        super(MaskBlock, self).__init__()
        self.depth = depth
        conv_layers, act_layers, beta_layers = [], [], []
        for i in range(depth):
            conv_layers.append(
                nn.Conv2d(
                    out_dim if i else in_dim,
                    out_dim,
                    kernel_size=3, stride=1, padding=1
                ),
            )
            act_layers.append(nn.PReLU(out_dim))
            beta_layers.append(nn.Parameter(torch.ones((1, out_dim, 1, 1)), requires_grad=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        self.act_layers = nn.Sequential(*act_layers)
        self.beta_layers = nn.ParameterList(beta_layers)

    def forward(self, x):
        for i in range(self.depth):
            if not i:
                x = self.act_layers[i](self.conv_layers[i](x) * self.beta_layers[i])  # different shape, cannot residual
            else:
                x = self.act_layers[i](self.conv_layers[i](x) * self.beta_layers[i] + x)
        return x[:, : 1, ...], x[:, 1: 4, ...]   # mask, refine


class TimeBlock(MaskBlock):
    def forward(self, x):   # (B, in_c, H, W)
        for i in range(self.depth):
            if not i:
                x = self.act_layers[i](self.conv_layers[i](x) * self.beta_layers[i])  # different shape, cannot residual
            else:
                x = self.act_layers[i](self.conv_layers[i](x) * self.beta_layers[i] + x)
        return x    # flow_delta


class MVFB(nn.Module):
    def __init__(self, in_planes, num_feature):
        super(MVFB, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, num_feature//2, 3, 2, 1),
            conv(num_feature//2, num_feature, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
        )
        self.conv_sq = conv(num_feature, num_feature//4)
        self.conv1 = nn.Sequential(
            conv(in_planes, 8, 3, 2, 1),
        )
        self.convblock1 = nn.Sequential(
            conv(8, 8),
        )
        self.lastconv = nn.ConvTranspose2d(num_feature//4 + 8, 4 + 1 + 3, 4, 2, 1)

    def forward(self, x, flow, scale):
        x0 = x
        flow0 = flow
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        x = torch.cat((x, flow), 1)
        x1 = self.conv0(x)
        x2 = self.conv_sq(self.convblock(x1) + x1)
        x2 = F.interpolate(x2, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        x3 = self.conv1(torch.cat((x0, flow0), 1))
        x4 = self.convblock1(x3)
        tmp = self.lastconv(torch.cat((x2, x4), dim=1))
        flow = tmp[:, : 4]
        mask = tmp[:, 4: 5]
        refine = tmp[:, 5: 8]
        return flow, mask, refine


class DMVFN(nn.Module):
    def __init__(self):
        super(DMVFN, self).__init__()
        self.convlstm = ConvLSTM(
            input_dim=3,
            hidden_dim=[16, 8, 3],
            kernel_size=(3, 3),
            num_layers=3,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.encoder = Head(8 * 2)
        self.block0 = MVFB(11 * 4 + 4, num_feature=160)
        self.block1 = MVFB(11 * 4 + 4, num_feature=160)
        self.block2 = MVFB(11 * 4 + 4, num_feature=160)
        self.block3 = MVFB(11 * 4 + 4, num_feature=80)
        self.block4 = MVFB(11 * 4 + 4, num_feature=80)
        self.block5 = MVFB(11 * 4 + 4, num_feature=80)
        self.block6 = MVFB(11 * 4 + 4, num_feature=44)
        self.block7 = MVFB(11 * 4 + 4, num_feature=44)
        self.block8 = MVFB(11 * 4 + 4, num_feature=44)
        self.bake_time = TimeBlock(2 + 1, 2)

    def forward(self, x, t_pred, scale, training=True):
        batch_size, _, height, width = x.shape  # 8, 9, H, W
        img0, img1, img2 = x[:, : 3], x[:, 3: 6], x[:, 6: 9]
        fea0, fea1, fea2 = self.encoder(img0), self.encoder(img1), self.encoder(img2)
        fea0 = torch.cat((fea0, img0), dim=1)
        fea1 = torch.cat((fea1, img1), dim=1)
        fea2 = torch.cat((fea2, img2), dim=1)
        warped_fea0, warped_fea1 = fea0, fea1
        merged_ls = []
        mask_ls = []
        refine_ls = []
        flow = Variable(torch.zeros(batch_size, 4, height, width)).cuda()
        stu = [
            self.block0, self.block1, self.block2, self.block3,
            self.block4, self.block5, self.block6, self.block7,
            self.block8
        ]
        for i in range(9):
            flow_d, mask, refine = stu[i](
                torch.cat(
                    (fea0, fea1, warped_fea0, warped_fea1), 1
                ),
                flow,
                scale=scale[i]
            )
            flow = flow + flow_d
            flow_delta_0 = self.bake_time(
                torch.cat(
                    (
                        flow[:, : 2].clone(),
                        (torch.ones(batch_size, 1, height, width) * t_pred).to(device)
                    ),
                    dim=1
                )
            )
            flow_delta_1 = self.bake_time(
                torch.cat(
                    (
                        flow[:, 2: 4].clone(),
                        (torch.ones(batch_size, 1, height, width) * (t_pred - 1)).to(device)
                    ),
                    dim=1
                )
            )
            flow = flow + torch.cat((flow_delta_0, flow_delta_1), dim=1)
            warped_fea0 = warp(fea0, flow[:, : 2])
            warped_fea1 = warp(fea1, flow[:, 2: 4])
            warped_img0 = warp(img0, flow[:, : 2])
            warped_img1 = warp(img1, flow[:, 2: 4])
            merged_ls.append(
                (warped_img0, warped_img1)
            )
            mask_ls.append(torch.sigmoid(mask))
            refine_ls.append(torch.tanh(refine))
        for i in range(9):
            merged_ls[i] = merged_ls[i][0] * mask_ls[i] + merged_ls[i][1] * (1 - mask_ls[i])
            merged_ls[i] += refine_ls[i]
            merged_ls[i] = torch.clamp(merged_ls[i], 0, 1)
        return merged_ls


if __name__ == '__main__':
    net = DMVFN(num_feature=64).cuda()
    x = torch.randn((2, 6, 64, 64)).cuda()
    y = net(x, scale=[4,4,4,2,2,2,1,1,1])
    print(y.shape)
