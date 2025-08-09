# ------------------------------------------------------------- #
# End-to-end one-stage with global and local information        #
# Functions for building model                                  #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter


class Backbone(nn.Module):
    def __init__(self, return_interm_layers: bool):
        super().__init__()
        
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]

        self.body = IntermediateLayerGetter(self.backbone, return_layers=return_layers)

    def forward(self, input):
        xs = self.body(input)

        feat = []
        for name, x in xs.items():
            feat.append(x)

        return feat
    

class DetectionHead(nn.Module):
    def __init__(self, d_feat, d_model):
        super().__init__()

        self.input_proj = nn.Conv2d(in_channels=d_feat, out_channels=d_model, kernel_size=1, stride=1, padding=0)

        # Global information        
        self.g_inside_slot = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=3, padding=1)
        self.g_junc1 = nn.Conv2d(in_channels=d_model, out_channels=2, kernel_size=3, padding=1)
        self.g_junc2 = nn.Conv2d(in_channels=d_model, out_channels=2, kernel_size=3, padding=1)
        self.g_slot_type = nn.Conv2d(in_channels=d_model, out_channels=3, kernel_size=3, padding=1)
        self.g_slot_occ = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=3, padding=1)

        # Local information
        self.l_junc_conf = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=3, padding=1)
        self.l_junc_loc = nn.Conv2d(in_channels=d_model, out_channels=2, kernel_size=3, padding=1)
        self.l_junc_ori = nn.Conv2d(in_channels=d_model, out_channels=2, kernel_size=3, padding=1)    


    def forward(self, feat):
        src = self.input_proj(feat)
        inside_slot = F.sigmoid(self.g_inside_slot(src))
        junc1 = F.tanh(self.g_junc1(src))
        junc2 = F.tanh(self.g_junc2(src))
        slot_type = F.softmax(self.g_slot_type(src), dim=1)
        slot_occ = F.sigmoid(self.g_slot_occ(src))

        junc_conf = F.sigmoid(self.l_junc_conf(src))
        junc_loc = F.sigmoid(self.l_junc_loc(src))
        junc_ori = F.tanh(self.l_junc_ori(src))

        output = torch.cat([inside_slot, junc1, junc2, slot_type, slot_occ, junc_conf, junc_loc, junc_ori], dim=1)

        return output
    


class Model(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.backbone = Backbone(return_interm_layers=False)
        d_feat = self.backbone.num_channels[-1]
        self.detection_head = DetectionHead(d_feat, d_model)

    def forward(self, img_input):
        feats = self.backbone(img_input)

        output = self.detection_head(feats[-1])

        return output.permute(0, 2, 3, 1)