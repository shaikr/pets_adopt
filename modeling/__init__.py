# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from torch import nn

from layers.fusion_layers import RegressionResnet18, ConcatModel, DenseFusionModel, DenseBeforeFusion, IgnoreNet, \
    DenseBeforeAndAfterFusion, DenseBeforeAndAfterFusionFromEmbeddings, IgnoreNetEmbeddings, ULTRAGODLIKEMODEL
from .example_model import ResNet18, RegNet
from torchvision.models.resnet import resnet152, resnet18


def get_embs_dims(cat_sz):
    return [(c, min(50, (c + 1) // 2)) for c in cat_sz]


def build_model(cfg):
    # model = ResNet18(1) #cfg.MODEL.NUM_CLASSES)
    # model_ft = resnet18(pretrained=True, progress=True)
    # set_parameter_requires_grad(model_ft, False)
    # num_features = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_features, 1)
    # model_ft=ConcatModel(512,142)
    # model_ft = DenseBeforeFusion(512, 142)
    # model_ft = IgnoreNetEmbeddings(512, 367)
    # model_ft = DenseBeforeAndAfterFusion(512, 142)

    model_ft = ULTRAGODLIKEMODEL(512,500,get_embs_dims([2, 160, 120, 2, 7, 7, 6, 4, 3, 3, 3, 3, 3, 14]), 10, 0.04, 2,
                                 [1000, 500], [0.001, 0.01], y_range=None,
                                 classify=None)
    # model_ft = DenseBeforeAndAfterFusionFromEmbeddings(512, 367)
    # model_ft = IgnoreNet(512,142)
    # model= resnet152(pretrained=True, progress=True, num_classes=1)
    # model = RegNet()
    print(model_ft)
    return model_ft


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
