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

    def Model_Init(self):
        model_start_time = time.time()
        # 定义模型的基本不变属性
        BaseModelConfig = self.config['BaseModelConfig']
        model = DeepLab(BaseModelConfig[0:4].split(','))
        
        # 加载模型
        ckpt = torch.load(self.config['ckpt'],map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        model = model.cuda()

        # 计算模型加载时间
        model_end_time = time.time()
        model_load_time = model_end_time - model_start_time
        print(f'Model Load Finished! Model load time is {model_load_time}')

        # 计算图片的均差等
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])




    def main(self):
        # Start_time = time.time()
        while self.is_detect_cv:
            predict_start_time = time.time()
            frame = cv2.capture(0)
            image = Image.read(frame).convert('RGB')
            target = IMage.open(frame).convert('L')
            sample = {'image':image,'label':target}
            tensor_in = self.Model_Init.composed_transforms(sample)['image'].unsqueeze(0)
            model.eval()
            if self.is_cuda:
                tensor_in = tensor_in.cuda()
            with torch.no_grad():
                output = model(tensor_in)

            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3,1],1)[1].detach.cpu().numpy()),3,normalize=False,range=(0,255))
            # save_image(grid_image,args.in_path....format(name[0:-4])
            predict_end_time = time.time()
            predcit_each_time = prediction_end_time - prediction_start_time
            print(f'Predict each frame time is {predcit_each_time}')



        pass #一会儿写




        












