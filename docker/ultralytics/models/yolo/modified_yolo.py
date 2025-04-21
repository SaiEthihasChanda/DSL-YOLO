import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f_DWRB, SADown, LASPPF  # Import custom layers
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel


class Modified_YOLOv8(nn.Module):
    """ YOLOv8 modified with C2f_DWRB, SADown, and LASPPF """

    def __init__(self, num_classes=6):
        super().__init__()
        
        # Define custom backbone
        self.backbone = nn.Sequential(
            Conv(3, 32, 3, 1),  # Initial conv layer
            C2f_DWRB(32, 64),  
            SADown(64),  
            C2f_DWRB(64, 128),  
            SADown(128),
            C2f_DWRB(128, 256),
        )
        
        # Replace SPFF with LASPPF
        self.neck = LASPPF(256)

        # Detection Head
        self.head = nn.Sequential(
            Conv(256, 128, 1, 1),
            nn.Conv2d(128, num_classes, kernel_size=1)  # Final layer predicting class scores
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)


# Integrate into YOLO Framework
class Custom_YOLO(DetectionModel):
    """ Custom YOLO model integrating Modified_YOLOv8 """

    def __init__(self, cfg=None, ch=3, nc=6):
        super().__init__(cfg, ch, nc)
        self.model = Modified_YOLOv8(num_classes=nc)


if __name__ == "__main__":
    model = Custom_YOLO()
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print("Output Shape:", y.shape)
