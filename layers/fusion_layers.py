import torch

from torch import nn
from torchvision.models import resnet18


class RegressionResnet18(nn.Module):
    def __init__(self):
        super(RegressionResnet18, self).__init__()
        self.features = nn.Sequential(
            *(list(resnet18(pretrained=True, progress=True).children())[:-1])
        )
        # self.model = resnet18(pretrained=True, progress=True)
        # self.set_parameter_requires_grad(self.model, False)
        # self.model.fc=None
        # num_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_features, 1)

    def forward(self, *input, **kwargs):
        return self.features.forward(*input, **kwargs).squeeze()

    # def forward(self, *input, **kwargs):
    #     self.model.forward()

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


class ConcatModel(nn.Module):
    def __init__(self, vision_feature_size, ds_feature_size):
        super(ConcatModel, self).__init__()
        self.vision_model = RegressionResnet18()
        self.fcs = nn.Sequential(
            # nn.Linear(vision_feature_size + ds_feature_size, vision_feature_size + ds_feature_size),
            # nn.Linear(vision_feature_size + ds_feature_size, vision_feature_size + ds_feature_size),
            # nn.Linear(vision_feature_size + ds_feature_size, vision_feature_size + ds_feature_size)
        )
        self.output = nn.Linear(vision_feature_size + ds_feature_size, 1)
        # torch.nn.init.xavier_uniform(self.fcs.weight)
        # torch.nn.init.xavier_uniform(self.output.weight)

    def forward(self, vectors):
        img, ds_vector = vectors
        x = self.vision_model(img)
        x = torch.cat((x, ds_vector.float()), dim=1)
        x=self.fcs(x)
        x=self.output(x)
        return x


class DenseFusionModel(nn.Module):
    def __init__(self, ds_vector_size):
        super(DenseFusionModel, self).__init__()
        self.vision_model = RegressionResnet18()
        self.fc = nn.Linear(self.vision_model.out_features, ds_vector_size)

    def forward(self, img, ds_vector):
        x = self.vision_model(img)
        x = self.fc(x)
        x = x + ds_vector.float()
        return x

        # self.fc =
