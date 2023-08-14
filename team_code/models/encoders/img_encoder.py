
from torchvision import models
from torch import nn
import torch


class ImageEncoder(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, model, c_dim, normalize=True, pretrained=False):
        super().__init__()
        self.normalize = normalize
        self._model = getattr(models, model)(pretrained=pretrained)
        self._model.fc = nn.LazyLinear(c_dim)

    @staticmethod
    def normalize_imagenet(x):
        """ Normalize input images according to ImageNet standards.
        Args:
            x (tensor): input images
        """
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def forward(self, inputs):
        if self.normalize:
            inputs = self.normalize_imagenet(inputs)
        x = self._model(inputs)
        return x