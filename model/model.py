import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url



class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)



class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3,7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size ==7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0, sa_input=None):  
        if sa_input is None:  
            avg_out = torch.mean(x0, dim=1, keepdim=True)
            max_out, _ = torch.max(x0, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
        else:
            x = sa_input  
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class enchance_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(CBR(in_c, out_c, kernel_size=1, padding=0), channel_attention(out_c))
        self.c2 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=6, dilation=6), channel_attention(out_c))
        self.c3 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=12, dilation=12), channel_attention(out_c))
        self.c4 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=18, dilation=18), channel_attention(out_c))
        self.c5 = CBR(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = CBR(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = spatial_attention()
        

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc + xs)
        x = self.sa(x)
        return x








model_urls = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'   


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls,
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model




def resnet50(pretrained=True, progress=True, **kwargs):
    
    return _resnet(Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)







"""Decouple Layer"""


class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return f_fg, f_bg, f_uc


"""Conv_Upsample"""


class Conv_Upsample(nn.Module):
    def __init__(self, in_c):
        super(Conv_Upsample, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()

        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc



class ContrastAwareLocalAttention(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0., window_size=7):  
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size  
        self.proj = nn.Linear(dim, dim)  
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_interact = nn.Linear(2 * num_heads, num_heads)  

        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ln = nn.LayerNorm(dim)  
        self.bn = nn.BatchNorm2d(dim)  

        
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        self.pool_attention = CBR(2 * dim, dim, kernel_size=1, padding=0)
        
        
        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1, act=False)
        )
        self.input_residual = nn.Sequential(  
            CBR(in_c, dim//2, kernel_size=1, padding=0, act=False),
            CBR(dim//2, dim, kernel_size=3, padding=1, act=False)
        )
        self.residual_fuse = CBR(2 * dim, dim, kernel_size=1, padding=0, act=False)  

        
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1, act=False)
        )

        
        self.act = nn.GELU()

        
        self.self_attn_q = nn.Linear(dim, dim)
        self.self_attn_k = nn.Linear(dim, dim)
        self.self_attn_v = nn.Linear(dim, dim)
        self.self_attn_drop = nn.Dropout(attn_drop)
        self.self_attn_proj = nn.Linear(dim, dim)

    def forward(self, x, fg, bg):
        x_shortcut = self.input_residual(x)  
        x_main = self.input_cbr(x)          
        x = self.act(self.residual_fuse(torch.cat([x_main, x_shortcut], dim=1)))  

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.local_self_attention(x, H, W)  
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.bn(x)  

        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # (B, C, H, W)
        v_unfolded = self.unfold(v).reshape(
            B, self.num_heads, self.head_dim, self.kernel_size**2, -1
        ).permute(0, 1, 4, 3, 2)  # (B, num_heads, N, k^2, head_dim)

        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')  # (B, num_heads, N, k^2, k^2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')
        attn = torch.cat([attn_fg, attn_bg], dim=1)  # (B, 2*num_heads, N, k^2, k^2)
        attn = attn.permute(0, 2, 3, 4, 1)  
        attn = self.attn_interact(attn)  # (B, N, k², k², 2*num_heads) → (B, N, k², k², num_heads)
        attn = attn.permute(0, 4, 1, 2, 3) 
        attn = F.softmax(attn, dim=-1)  

        x_weighted = self.apply_attention(attn, v_unfolded, B, H, W, C)
        x_weighted = x_weighted.permute(0, 3, 1, 2)  # (B, C, H, W)

        out = self.output_cbr(x_weighted)
        out = self.act(out + x_shortcut)  
        return out

    def local_self_attention(self, x, H, W):
        B, H, W, C = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim
        window_size = self.window_size

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        x_padded = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # (B, H+pad_h, W+pad_w, C)
        H_pad, W_pad = x_padded.shape[1], x_padded.shape[2]

        x_win = x_padded.reshape(
            B, H_pad // window_size, window_size,
            W_pad // window_size, window_size, C
        ).permute(0, 1, 3, 2, 4, 5).reshape(
            B, -1, window_size * window_size, C  # (B, num_windows, win_size^2, C)
        )

        q = self.self_attn_q(x_win).reshape(
            B, -1, window_size*window_size, num_heads, head_dim
        ).permute(0, 1, 3, 2, 4)  # (B, num_windows, num_heads, win_size^2, head_dim)
        k = self.self_attn_k(x_win).reshape(
            B, -1, window_size*window_size, num_heads, head_dim
        ).permute(0, 1, 3, 2, 4)
        v = self.self_attn_v(x_win).reshape(
            B, -1, window_size*window_size, num_heads, head_dim
        ).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.self_attn_drop(attn)

        x_win_attn = (attn @ v).permute(0, 1, 3, 2, 4).reshape(
            B, -1, window_size*window_size, C
        )

        x_win_attn = x_win_attn.reshape(
            B, H_pad // window_size, W_pad // window_size,
            window_size, window_size, C
        ).permute(0, 1, 3, 2, 4, 5).reshape(B, H_pad, W_pad, C)
        x_self = x_win_attn[:, :H, :W, :]  
        x_self = self.self_attn_proj(x_self)  
        return x_self

    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        
        feature_map_avg = self.pool(feature_map.permute(0, 3, 1, 2))
        feature_map_max = F.max_pool2d(
            feature_map.permute(0, 3, 1, 2), kernel_size=self.stride, 
            stride=self.stride, ceil_mode=True
        )
        feature_map_pooled = torch.cat([feature_map_avg, feature_map_max], dim=1)
        feature_map_pooled = self.pool_attention(feature_map_pooled).permute(0, 2, 3, 1)  # (B, h, w, dim)
        
        feature_map_pooled = self.ln(feature_map_pooled)
        
        attn = attn_layer(feature_map_pooled).reshape(
            B, h * w, self.num_heads, self.kernel_size**2, self.kernel_size**2
        ).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1
        )
        x_weighted = F.fold(
            x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
            padding=self.padding, stride=self.stride
        )
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted
        





class output_block(nn.Module):

    def __init__(self, in_c, out_c=1):
        super().__init__()

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.fuse=CBR(in_c*3,in_c, kernel_size=3, padding=1)
        self.c1 = CBR(in_c, 128, kernel_size=3, padding=1)
        self.c2 = CBR(128, 64, kernel_size=1, padding=0)
        self.c3 = nn.Conv2d(64, out_c, kernel_size=1, padding=0)
        self.sig=nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x2 = self.up_2x2(x2)
        x3 = self.up_4x4(x3)

        x = torch.cat([x1, x2, x3], axis=1)
        x=self.fuse(x)

        x=self.up_2x2(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x=self.sig(x)
        return x


class Preprocess(nn.Module):

    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x

class MixedAttention(nn.Module):
    def __init__(self, in_c, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = channel_attention(in_c, ratio)  
        self.sa = spatial_attention(kernel_size)  
        self.map = nn.Conv2d(in_c, 2, kernel_size=1, padding=0)  

    def forward(self, x):
        x_ca = self.ca(x)  
        sa_input = self.map(x_ca)
        x_sa = self.sa(x_ca, sa_input)  
        return x_sa



class DGDecoder(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.attn = nn.Sequential(
            nn.Conv2d(in_c + out_c, 1, kernel_size=1, padding=0),  
            nn.Sigmoid()  
        )
        self.c1 = CBR(in_c + out_c, out_c, kernel_size=1, padding=0)  
        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)
        self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention()
        self.mixed_attn = MixedAttention(out_c) 

    def forward(self, x, skip):
        x = self.up(x)
        fused = torch.cat([x, skip], axis=1)
        weight = self.attn(fused)  # shape: [B,1,H,W]
        x_weighted = x * weight
        skip_weighted = skip * (1 - weight)
        fused_weighted = torch.cat([x_weighted, skip_weighted], axis=1)
        
        x = self.c1(fused_weighted)  

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        x = self.ca(x)
        x = self.mixed_attn(x)
        return x       
        

class CALDGSeg(nn.Module):
    def __init__(self, H=256, W=256):
        super().__init__()

        self.H = H
        self.W = W
        

        """ Backbone: ResNet50 """
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # [batch_size, 256, h/4, w/4]
        self.layer2 = backbone.layer2  # [batch_size, 512, h/8, w/8]
        self.layer3 = backbone.layer3  # [batch_size, 1024, h/16, w/16]

        
        """ FEM """
        self.enchance1 = enchance_conv(64, 128)
        self.enchance2 = enchance_conv(256, 128)
        self.enchance3 = enchance_conv(512, 128)
        self.enchance4 = enchance_conv(1024, 128)

        """ Decouple Layer """
        self.decouple_layer = DecoupleLayer(1024, 128)

        """ Adjust the shape of decouple output """
        self.preprocess_fg4 = Preprocess(128, 128, 1)  # 1/16
        self.preprocess_bg4 = Preprocess(128, 128, 1)  # 1/16

        self.preprocess_fg3 = Preprocess(128, 128, 2)  # 1/8
        self.preprocess_bg3 = Preprocess(128, 128, 2)  # 1/8

        self.preprocess_fg2 = Preprocess(128, 128,4)  # 1/4
        self.preprocess_bg2 = Preprocess(128, 128, 4)  # 1/4

        self.preprocess_fg1 = Preprocess(128, 128,8)  # 1/2
        self.preprocess_bg1 = Preprocess(128, 128, 8)  # 1/2

        self.aux_head = Conv_Upsample(128)

        """ Contrast-Driven Feature Aggregation """
        self.up2X = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cala4 = ContrastAwareLocalAttention(128, 128, 4)
        self.cala3 = ContrastAwareLocalAttention(128 + 128, 128, 4)
        self.cala2 = ContrastAwareLocalAttention(128 + 128, 128, 4)
        self.cala1 = ContrastAwareLocalAttention(128 + 128, 128, 4)
        

        """ Decoder """
        self.decoder1 = DGDecoder(128, 128, scale=2)
        self.decoder2 = DGDecoder(128, 128, scale=2)
        self.decoder3 = DGDecoder(128, 128, scale=2)
        
        """ Output Block """
        self.output_block = output_block(128, 1)

    def forward(self, image):

        """ Backbone: ResNet50 """
        x0 = image
        x1 = self.layer0(x0)  ## [-1, 64, h/2, w/2]
        x2 = self.layer1(x1)  ## [-1, 256, h/4, w/4]
        x3 = self.layer2(x2)  ## [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)  ## [-1, 1024, h/16, w/16]

        """ Dilated Conv """
        d1 = self.enchance1(x1)
        d2 = self.enchance2(x2)
        d3 = self.enchance3(x3)
        d4 = self.enchance4(x4)

        """ Decouple Layer """
        f_fg, f_bg, f_uc = self.decouple_layer(x4)
        """ Auxiliary Head """
        mask_fg, mask_bg, mask_uc = self.aux_head(f_fg, f_bg, f_uc)

        """ Contrast-Driven Feature Aggregation """
        f_fg4 = self.preprocess_fg4(f_fg)
        f_bg4 = self.preprocess_bg4(f_bg)
        f_fg3 = self.preprocess_fg3(f_fg)
        f_bg3 = self.preprocess_bg3(f_bg)
        f_fg2 = self.preprocess_fg2(f_fg)
        f_bg2 = self.preprocess_bg2(f_bg)
        f_fg1 = self.preprocess_fg1(f_fg)
        f_bg1 = self.preprocess_bg1(f_bg)


        f4 = self.cala4(d4, f_fg4, f_bg4)
        f4_up = self.up2X(f4)
        f_4_3 = torch.cat([d3, f4_up], dim=1)
        f3 = self.cala3(f_4_3, f_fg3, f_bg3)
        f3_up = self.up2X(f3)
        f_3_2 = torch.cat([d2, f3_up], dim=1)
        f2 = self.cala2(f_3_2, f_fg2, f_bg2)
        f2_up = self.up2X(f2)
        f_2_1 = torch.cat([d1, f2_up], dim=1)
        f1 = self.cala1(f_2_1, f_fg1, f_bg1)
        

        """ Decoder """
        f_1 = self.decoder1(f2, f1)
        f_2 = self.decoder2(f3, f2)
        f_3 = self.decoder3(f4, f3)
        
        """ Output Block """
        mask = self.output_block(f_1, f_2, f_3)
       


        return mask, mask_fg, mask_bg, mask_uc



if __name__ == "__main__":
    model = ConDSeg().cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    output = model(input_tensor)
    print(output.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")  