import torch.nn as nn
from torchvision import transforms

class Reshape(nn.Module):
    def __init__(self, dimmensions):
        super().__init__()
        # Keep the first dimmension (batch size)
        shape = (-1, *dimmensions)
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
     
class Trim(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x[:, :, :self.target_shape[1], :self.target_shape[2]]
    
class Scale:
    def __init__(self, size=(140, 140)):
        self.size = size

    def __call__(self, image):
        return transforms.functional.resize(image, self.size)
