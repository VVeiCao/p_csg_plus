from torch import nn
from torchvision import models
import torch
class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        c_dim: output feature dimension
        in_channels: input channels
    """

    def __init__(self, model, c_dim, in_channels=2, pretrained=False):
        super().__init__()

        self._model =  getattr(models, model)(pretrained=pretrained)
        self._model.fc = nn.LazyLinear(c_dim)
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)

    def forward(self, inputs):
        x = self._model(inputs)
        return x