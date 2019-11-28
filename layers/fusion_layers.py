import torch

from torch import nn
from torchvision.models import resnet18


class RegressionResnet18(nn.Module):
    def __init__(self):
        super(RegressionResnet18, self).__init__()
        self.features = nn.Sequential(
            *(list(resnet18(pretrained=False, progress=True).children())[:-1])
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
        #     nn.Linear(vision_feature_size + ds_feature_size, vision_feature_size + ds_feature_size),
        #     nn.ReLU(),
        #     nn.Linear(vision_feature_size + ds_feature_size, vision_feature_size + ds_feature_size),
        #     nn.ReLU(),
        #     nn.Linear(vision_feature_size + ds_feature_size, vision_feature_size + ds_feature_size),
        #     nn.ReLU()
        )
        self.output = nn.Linear(vision_feature_size + ds_feature_size, 1)
        # torch.nn.init.xavier_uniform(self.fcs.weight)
        torch.nn.init.xavier_uniform(self.output.weight)
        [torch.nn.init.xavier_uniform(layer.weight) for layer in list(self.fcs.modules()) if type(layer) == nn.Linear]

    def forward(self, vectors):
        img, ds_vector = vectors
        x = self.vision_model(img)
        x = torch.cat((x, ds_vector.float()), dim=1)
        x = self.fcs(x)
        x = self.output(x)
        return x


class DenseFusionModel(nn.Module):
    def __init__(self, vision_feature_size, ds_feature_size):
        super(DenseFusionModel, self).__init__()
        self.vision_model = RegressionResnet18()
        self.fusion = nn.Linear(vision_feature_size, ds_feature_size)
        self.fcs = nn.Sequential(
            # nn.Linear(ds_feature_size, ds_feature_size),
            # nn.ReLU(),
            # nn.Linear(ds_feature_size, ds_feature_size),
            # nn.ReLU(),
            # nn.Linear(ds_feature_size, ds_feature_size),
            # nn.ReLU()
        )
        self.output = nn.Linear(ds_feature_size, 1)
        torch.nn.init.xavier_uniform(self.output.weight)
        torch.nn.init.xavier_uniform(self.fusion.weight)
        [torch.nn.init.xavier_uniform(layer.weight) for layer in list(self.fcs.modules()) if type(layer) == nn.Linear]

    def forward(self, vectors):
        img, ds_vector = vectors
        x = self.vision_model(img)
        x = self.fusion(x)
        x = x + ds_vector.float()
        x = self.fcs(x)
        x = self.output(x)
        return x


class Test(nn.Module):
    def __init__(self, vision_feature_size, ds_feature_size):
        super(Test, self).__init__()
        self.vision_model = RegressionResnet18()
        self.fusion = nn.Linear(vision_feature_size, ds_feature_size)
        # self.ds_embedding_network = RegressionResnet18()
        self.fcs = nn.Sequential(
            nn.Linear(ds_feature_size, ds_feature_size),
            nn.ReLU(),
            nn.Linear(ds_feature_size, ds_feature_size),
            nn.ReLU(),
            nn.Linear(ds_feature_size, vision_feature_size),
            nn.ReLU()
        )
        self.output = nn.Linear(vision_feature_size, 1)
        # torch.nn.init.xavier_uniform(self.output.weight)
        # torch.nn.init.xavier_uniform(self.fusion.weight)
        [torch.nn.init.xavier_uniform(layer.weight) for layer in list(self.fcs.modules()) if type(layer) == nn.Linear]

    def forward(self, vectors):
        img, ds_vector = vectors
        ds_vector = ds_vector.float()
        x = self.vision_model(img)
        x_tag = self.fcs(ds_vector)
        # x = self.fusion(x)
        x = x + x_tag
        # x = self.fcs(x)
        x = self.output(x)
        return x


class Test2(nn.Module):
    def __init__(self, vision_feature_size, ds_feature_size):
        super(Test2, self).__init__()
        self.vision_model = RegressionResnet18()
        self.fusion = nn.Linear(vision_feature_size, ds_feature_size)
        # self.ds_embedding_network = RegressionResnet18()
        self.fcs1 = nn.Sequential(
            nn.Linear(ds_feature_size, ds_feature_size),
            nn.ReLU(),
            nn.Linear(ds_feature_size, ds_feature_size),
            nn.ReLU(),
            nn.Linear(ds_feature_size, vision_feature_size),
            nn.ReLU()
        )

        self.fcs2 = nn.Sequential(
            nn.Linear(vision_feature_size, vision_feature_size),
            nn.ReLU(),
            nn.Linear(vision_feature_size, vision_feature_size),
            nn.ReLU()
        )

        self.output = nn.Linear(vision_feature_size, 1)
        # torch.nn.init.xavier_uniform(self.output.weight)
        # torch.nn.init.xavier_uniform(self.fusion.weight)
        [torch.nn.init.xavier_uniform(layer.weight) for layer in list(self.fcs1.modules()) if type(layer) == nn.Linear]
        [torch.nn.init.xavier_uniform(layer.weight) for layer in list(self.fcs2.modules()) if type(layer) == nn.Linear]

    def forward(self, vectors):
        img, ds_vector = vectors
        ds_vector = ds_vector.float()
        x = self.vision_model(img)
        x_tag = self.fcs1(ds_vector)
        x = x + x_tag
        x = self.fcs2(x)
        x = self.output(x)
        return x


class IgnoreNet(nn.Module):
    def __init__(self, vision_feature_size, ds_feature_size):
        super(IgnoreNet, self).__init__()
        self.vision_model = RegressionResnet18()
        self.output = nn.Linear(vision_feature_size, 1)
        # torch.nn.init.xavier_uniform(self.output.weight)

    def forward(self, vectors):
        img, ds_vector = vectors
        x = self.vision_model(img)
        x = self.output(x)
        return x
