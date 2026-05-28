import torch as t
import torch.nn as nn
import torch.nn.functional as F
from deel import torchlip
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d, AdaptiveOrthoConvTranspose2d
from orthogonium.reparametrizers import DEFAULT_ORTHO_PARAMS

#################################
# ## TOY NETWORK FOR 2D DATA ## #
#################################

class ToyNetLinLip(nn.Module):
    def __init__(self, dim=2, n_f=32, leak=0.05):
        super(ToyNetLinLip, self).__init__()
        self.f = nn.Sequential(
            torchlip.SpectralLinear(dim, n_f),
            nn.LeakyReLU(leak),
            torchlip.SpectralLinear(n_f, n_f * 2),
            nn.LeakyReLU(leak),
            torchlip.SpectralLinear(n_f * 2, n_f * 2),
            nn.LeakyReLU(leak),
            torchlip.SpectralLinear(n_f * 2, n_f * 2),
            nn.LeakyReLU(leak),
            torchlip.SpectralLinear(n_f * 2, 1))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.f(x).squeeze(-1)

class ToyNetLin(nn.Module):
    def __init__(self, dim=2, n_f=32, leak=0.05):
        super(ToyNetLin, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(dim, n_f),
            nn.LeakyReLU(leak),
            nn.Linear(n_f, n_f * 2),
            nn.LeakyReLU(leak),
            nn.Linear(n_f * 2, n_f * 2),
            nn.LeakyReLU(leak),
            nn.Linear(n_f * 2, n_f * 2),
            nn.LeakyReLU(leak),
            nn.Linear(n_f * 2, 1))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.f(x).squeeze(-1)




class ToyNet(nn.Module):
    def __init__(self, dim=2, n_f=32, leak=0.05):
        super(ToyNet, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(dim, n_f, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, 1, 1, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze()


#########################
# ## VANILLA CONVNET ## #
#########################

class VanillaNet(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05):
        super(VanillaNet, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*8, 1, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze()

class VanillaNetHalfLip(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05):
        super(VanillaNetHalfLip, self).__init__()
        self.f = nn.Sequential(
            torchlip.SpectralConv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*8, 1, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze()
        
class VanillaNetLLip2(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, Lip_const=10): # TODO
        super(VanillaNetLLip2, self).__init__()
        self.Lip_const = Lip_const
        self.f = nn.Sequential(
            torchlip.SpectralConv2d(n_c, n_f, 3, 1, 1),
            torchlip.GroupSort2(),
            torchlip.SpectralConv2d(n_f, n_f * 2, 5, 2, 2),
            torchlip.GroupSort2(),
            torchlip.SpectralConv2d(n_f*2, n_f*4, 5, 2, 2),
            torchlip.GroupSort2(),
            torchlip.SpectralConv2d(n_f*4, n_f*8, 5, 2, 2),
            torchlip.GroupSort2(),
            torchlip.SpectralConv2d(n_f*8, 1, 4, 1, 0))

    def forward(self, x):
        return self.Lip_const * self.f(x).squeeze()

class VanillaNet2(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05):
        super(VanillaNet2, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            torchlip.GroupSort2(),
            nn.Conv2d(n_f, n_f * 2, 5, 2, 2),
            torchlip.GroupSort2(),
            nn.Conv2d(n_f*2, n_f*4, 5, 2, 2),
            torchlip.GroupSort2(),
            nn.Conv2d(n_f*4, n_f*8, 5, 2, 2),
            torchlip.GroupSort2(),
            nn.Conv2d(n_f*8, 1, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze()

class VanillaNetLLip(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, Lip_const=10):
        super(VanillaNetLLip, self).__init__()
        self.Lip_const = Lip_const
        self.f = nn.Sequential(
            torchlip.SpectralConv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak),
            torchlip.SpectralConv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(leak),
            torchlip.SpectralConv2d(n_f*2, n_f*4, 4, 2, 1),
            nn.LeakyReLU(leak),
            torchlip.SpectralConv2d(n_f*4, n_f*8, 4, 2, 1),
            nn.LeakyReLU(leak),
            torchlip.SpectralConv2d(n_f*8, 1, 4, 1, 0))

    def forward(self, x):
        return self.Lip_const * self.f(x).squeeze()    

class PixelUnshuffle2x(nn.Module):
    def forward(self, x):
        return nn.functional.pixel_unshuffle(x, 2)

class VanillaNetOrthogonium(nn.Module):
    def __init__(self, n_c=1, n_f=32, Lip_const=1):
        super(VanillaNetOrthogonium, self).__init__()
        self.Lip_const = Lip_const

        self.f = nn.Sequential(
            PixelUnshuffle2x(),   # 1x32x32 -> 4x16x16
            PixelUnshuffle2x(),   # 4x16x16 -> 16x8x8
            PixelUnshuffle2x(),   # 16x8x8 -> 64x4x4
            PixelUnshuffle2x(),   # 64x4x4 -> 256x2x2

            AdaptiveOrthoConv2d(256,64,3,padding="same",padding_mode="circular"),
            torchlip.GroupSort2(),

            AdaptiveOrthoConv2d(64,64,5,padding="same",padding_mode="circular"),
            torchlip.GroupSort2(),

            AdaptiveOrthoConv2d(64,64,5,padding="same",padding_mode="circular"),
            torchlip.GroupSort2(),

            AdaptiveOrthoConv2d(64,64,5,padding="same",padding_mode="circular"),
            torchlip.GroupSort2(),

            AdaptiveOrthoConv2d(64,64,5,padding="same",padding_mode="circular"),
            torchlip.GroupSort2(),


            nn.Flatten(),
            torchlip.SpectralLinear(256, 1),
        )

    def forward(self, x):
        return self.Lip_const * self.f(x).squeeze()


class VanillaNetOrthogoniumCifar(nn.Module):
    def __init__(self, n_c=3, n_f=32, Lip_const=1):
        super(VanillaNetOrthogoniumCifar, self).__init__()
        self.Lip_const = Lip_const

        self.f = nn.Sequential(
            PixelUnshuffle2x(),   # 3x32x32 -> 12x16x16
            PixelUnshuffle2x(),   # 12x16x16 -> 48x8x8
            PixelUnshuffle2x(),   # 48x8x8 -> 192x4x4

            AdaptiveOrthoConv2d(192,192,3,padding="same",padding_mode="circular"),
            torchlip.GroupSort2(),

            AdaptiveOrthoConv2d(192,48,5,padding="same",padding_mode="circular"),
            torchlip.GroupSort2(),

            AdaptiveOrthoConv2d(48,48,5,padding="same",padding_mode="circular"),
            torchlip.GroupSort2(),


            nn.Flatten(),
            torchlip.SpectralLinear(768, 1),
        )

    def forward(self, x):
        return self.Lip_const * self.f(x).squeeze()

######################
# ## NONLOCAL NET ## #
######################
# implementation with minor changes from https://github.com/AlexHex7/Non-local_pytorch
# Original Version: Copyright (c) 2018 AlexHex7

class NonlocalNet(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05):
        super(NonlocalNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=n_c, out_channels=n_f, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),

            NonLocalBlock(in_channels=n_f),
            nn.Conv2d(in_channels=n_f, out_channels=n_f * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),

            NonLocalBlock(in_channels=n_f * 2),
            nn.Conv2d(in_channels=n_f * 2, out_channels=n_f * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(n_f * 4) * 4 * 4, out_features=n_f * 8),
            nn.LeakyReLU(negative_slope=leak),
            nn.Linear(in_features=n_f * 8, out_features=1)
        )

    def forward(self, x):
        conv_out = self.convs(x).view(x.shape[0], -1)
        return self.fc(conv_out).squeeze()

# structure of non-local block (from Non-Local Neural Networks https://arxiv.org/abs/1711.07971)
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, sub_sample=True):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // 2)

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = t.matmul(theta_x, phi_x)
        f_div_c = F.softmax(f, dim=-1)

        y = t.matmul(f_div_c, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        w_y = self.W(y)
        z = w_y + x

        return z


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm(n_filters, norm):
    if norm is None:
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm=None, leak=0.2):
        super().__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1   = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2   = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.conv1(self.lrelu(self.bn1(x)))
        out = self.conv2(self.lrelu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=2, n_c=3, norm=None, leak=0.2, n_classes=10):
        super().__init__()
        self.lrelu    = nn.LeakyReLU(leak)
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1  = conv3x3(n_c, nStages[0])
        self.layer1 = self._wide_layer(nStages[1], n, stride=1, norm=norm, leak=leak)
        self.layer2 = self._wide_layer(nStages[2], n, stride=2, norm=norm, leak=leak)
        self.layer3 = self._wide_layer(nStages[3], n, stride=2, norm=norm, leak=leak)
        self.bn1    = get_norm(nStages[3], norm)
        self.linear = nn.Linear(nStages[3], n_classes)

    def _wide_layer(self, planes, num_blocks, stride, norm, leak):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(wide_basic(self.in_planes, planes, stride=s, norm=norm, leak=leak))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits.logsumexp(1)          # scalar energy per sample → (B,)



class Wide_ResNetSoftPlus(nn.Module):
    def __init__(self, depth=16, widen_factor=2, n_c=3, norm=None, leak=0.2, n_classes=10):
        super().__init__()
        self.lrelu    = nn.LeakyReLU(leak)
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1  = conv3x3(n_c, nStages[0])
        self.layer1 = self._wide_layer(nStages[1], n, stride=1, norm=norm, leak=leak)
        self.layer2 = self._wide_layer(nStages[2], n, stride=2, norm=norm, leak=leak)
        self.layer3 = self._wide_layer(nStages[3], n, stride=2, norm=norm, leak=leak)
        self.bn1    = get_norm(nStages[3], norm)
        self.linear = nn.Linear(nStages[3], n_classes)
        self.softplus = nn.Softplus()

    def _wide_layer(self, planes, num_blocks, stride, norm, leak):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(wide_basic(self.in_planes, planes, stride=s, norm=norm, leak=leak))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return self.softplus(logits).logsumexp(1)          # scalar energy per sample → (B,)










# lip

def conv3x3_lip(in_planes, out_planes, stride=1):
    return torchlip.SpectralConv2d(in_planes, out_planes, kernel_size=3, 
                                   stride=stride, padding=1, bias=True)

class wide_basic_lip(nn.Module):
    def __init__(self, in_planes, planes, stride=1, leak=0.2):
        super().__init__()
        #self.bn1 = torchlip.BatchCentering(in_planes)
        self.bn1 = Identity()
        self.lip_residual=torchlip.LipResidual()

        self.act = torchlip.GroupSort2()
        self.conv1 = torchlip.SpectralConv2d(in_planes, planes, kernel_size=3, 
                                             padding=1, bias=True)
        
        #self.bn2 = torchlip.BatchCentering(planes)
        self.bn2 = Identity()
        self.conv2 = torchlip.SpectralConv2d(planes, planes, kernel_size=3, 
                                             stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                torchlip.SpectralConv2d(in_planes, planes, kernel_size=1, 
                                        stride=stride, padding=0, bias=True),
            )

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))    
        out = self.conv2(self.act(self.bn2(out))) 
        out = (out + self.lip_residual(out, self.shortcut(x)))/2
        return out

class Wide_ResNet_LLip(nn.Module):
    def __init__(self, depth=16, widen_factor=2, n_c=3, n_classes=10, Lip_const = 1.0):
        super().__init__()
        self.Lip_const = Lip_const

        self.activation = torchlip.GroupSort2()
        self.lipavgpool2d = torchlip.ScaledAvgPool2d(kernel_size=8)
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16*k, 32*k, 64*k]

        # Première couche
        self.conv1 = conv3x3_lip(n_c, n_stages[0])
        
        # Couches Wide ResNet
        self.layer1 = self._wide_layer(n_stages[1], n, stride=1)
        self.layer2 = self._wide_layer(n_stages[2], n, stride=2)
        self.layer3 = self._wide_layer(n_stages[3], n, stride=2)
        
        #self.bn1 = torchlip.BatchCentering(n_stages[3])
        self.bn1 = Identity()
        
        self.linear = torchlip.SpectralLinear(n_stages[3], n_classes)

    def _wide_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(wide_basic_lip(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.bn1(out)
        out = self.activation(out)
        
        
        out = self.lipavgpool2d(out) 
        
        out = out.view(out.size(0), -1)
        
        logits = self.linear(out)
        return self.Lip_const * logits.logsumexp(1)

class Wide_ResNet_LLipSoftPlus(nn.Module):
    def __init__(self, depth=16, widen_factor=2, n_c=3, n_classes=10, Lip_const = 1.0):
        super().__init__()
        self.Lip_const = Lip_const

        self.activation = torchlip.GroupSort2()
        self.lipavgpool2d = torchlip.ScaledAvgPool2d(kernel_size=8)
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16*k, 32*k, 64*k]

        # Première couche
        self.conv1 = conv3x3_lip(n_c, n_stages[0])
        
        # Couches Wide ResNet
        self.layer1 = self._wide_layer(n_stages[1], n, stride=1)
        self.layer2 = self._wide_layer(n_stages[2], n, stride=2)
        self.layer3 = self._wide_layer(n_stages[3], n, stride=2)
        
        #self.bn1 = torchlip.BatchCentering(n_stages[3])
        self.bn1 = Identity()
        
        self.linear = torchlip.SpectralLinear(n_stages[3], n_classes)
        self.softplus = nn.Softplus()

    def _wide_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(wide_basic_lip(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.bn1(out)
        out = self.activation(out)
        
        
        out = self.lipavgpool2d(out) 
        
        out = out.view(out.size(0), -1)
        
        logits = self.linear(out)
        return self.Lip_const * self.softplus(logits).logsumexp(1)