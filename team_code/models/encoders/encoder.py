import torch
from torch import nn
from .img_encoder import ImageEncoder
from .lidar_encoder import LidarEncoder


class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(model=config.image_encoder_model, c_dim=config.c_dim, pretrained=config.pretrained)
        self.lidar_encoder = LidarEncoder(model=config.lidar_encoder_model, c_dim=config.c_dim, in_channels=2, pretrained=config.pretrained)

        
    def forward(self, image_inputs, lidar_inputs, measurements):
        """
            Args:
            image_inputs: image input with the shape of [B,C,H,W]
            lidar_inputs: lidar input with the shape of [B,C,H,W]
            measurements: some measurements of current states. 
        """
        image_inputs = image_inputs / 255.0 # try to fix this
        image_features = self.image_encoder(image_inputs)
        lidar_features = self.lidar_encoder(lidar_inputs)

        fused_features = torch.cat([image_features, lidar_features, measurements], dim=1)

        return fused_features
