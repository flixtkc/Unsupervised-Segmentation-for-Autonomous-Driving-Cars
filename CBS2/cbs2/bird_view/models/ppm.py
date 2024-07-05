import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        print('\n---------------------WITH PPM-----------------------\n')
        super(PPM, self).__init__()
        self.features = []
        self.in_dim = in_dim  # Store in_dim as an instance variable
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))

        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for i, f in enumerate(self.features):
            feature_out = f(x)
            feature_out_upsampled = F.interpolate(feature_out, x_size[2:], mode='bilinear', align_corners=True)
            out.append(feature_out_upsampled)
        concatenated_output = torch.cat(out, 1)
        return concatenated_output
