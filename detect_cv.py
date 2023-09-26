import argparse
import os
import numpy as np
import time

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image


class detect_cv(self):
    def __init__(self):
        self.is_detect = True
        self.is_detect_cv = True
        self.is_cuda = True
        with open("./config.yaml",encoding="utf8") as f:
            self.config = yaml.safe_load(f)
        self.num_classes
        self.model = DeepLab(num_classes=num_classes,
            backbone=self.backbone,
            output_stride = args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn
        )
    def Model_Init():
        pass

    def main():
        pass #一会儿写




        












