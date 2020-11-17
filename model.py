# 
#   Deep Scene
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import Tensor
from torch.jit import export
from torch.nn import Dropout, Linear, Module, Sequential
from torch.nn.functional import interpolate
from torchvision.models import mobilenet_v2
from torchsummary import summary

class DeepScene (Module):
    """
    Scene classifier for conditional image editing.

    Parameters:
        classes (int): Number of output classes.
    """

    def __init__ (self, classes=2):
        super(DeepScene, self).__init__()
        self.backbone = mobilenet_v2(pretrained=True)
        self.backbone.classifier = Sequential(
            Dropout(0.2),
            Linear(self.backbone.last_channel, classes)
        )

    def forward (self, input: Tensor) -> Tensor:
        input = interpolate(input, (512, 512), mode="bilinear", align_corners=False)
        logits = self.backbone(input)
        return logits


if __name__ == "__main__":
    model = DeepScene(classes=2)
    summary(model, (3, 1024, 1024), batch_size=8)