import torch

from torch import nn
from torch.nn.init import kaiming_normal
from torchvision.models import resnet18

from modeling.embedding_model import EmbeddingModel

EmbeddingModel


class RegressionResnet18(nn.Module):
    def __init__(self):
        super(RegressionResnet18, self).__init__()
        self.features = nn.Sequential(
            *(list(resnet18(pretrained=False, progress=True).children())[:-1])
        )

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


class DenseBeforeFusion(nn.Module):
    def __init__(self, vision_feature_size, ds_feature_size):
        super(DenseBeforeFusion, self).__init__()
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


class DenseBeforeAndAfterFusion(nn.Module):
    def __init__(self, vision_feature_size, ds_feature_size):
        super(DenseBeforeAndAfterFusion, self).__init__()
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


class DenseBeforeAndAfterFusionFromEmbeddings(nn.Module):
    def __init__(self, vision_feature_size, ds_feature_size):
        super(DenseBeforeAndAfterFusionFromEmbeddings, self).__init__()
        self.fusion = nn.Linear(vision_feature_size, ds_feature_size)
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
        [torch.nn.init.xavier_uniform(layer.weight) for layer in list(self.fcs1.modules()) if type(layer) == nn.Linear]
        [torch.nn.init.xavier_uniform(layer.weight) for layer in list(self.fcs2.modules()) if type(layer) == nn.Linear]

    def forward(self, vectors):
        image_embedding, ds_vector = vectors
        ds_vector = ds_vector.float()
        x_tag = self.fcs1(ds_vector)
        x = image_embedding + x_tag
        x = self.fcs2(x)
        x = self.output(x)
        return x


class IgnoreNetEmbeddings(nn.Module):
    def __init__(self, vision_feature_size, ds_feature_size):
        super(IgnoreNetEmbeddings, self).__init__()
        self.ds_feature_size = ds_feature_size
        self.vision_feature_size = vision_feature_size

        self.fc = nn.Linear(vision_feature_size, 1)
        torch.nn.init.xavier_uniform(self.fc.weight)

    def forward(self, vectors):
        image_embedding, ds_vector = vectors
        x = self.fc(image_embedding)
        return x


class ULTRAGODLIKEMODEL(nn.Module):
    def __init__(self, vision_feature_size, ds_feature_size, emb_szs, n_cont, emb_drop, out_sz, szs, drops,
                 y_range=None, use_bn=False,
                 classify=None):
        super(ULTRAGODLIKEMODEL, self).__init__()
        self.vision_model = RegressionResnet18()
        self.embedding_model = EmbeddingModel(emb_szs=emb_szs, n_cont=n_cont, emb_drop=emb_drop, out_sz=out_sz, szs=szs,
                                              drops=drops, y_range=y_range, use_bn=False, classify=None)

        self.dim_alignment = nn.Sequential(
            nn.Linear(ds_feature_size, ds_feature_size),
            nn.ReLU(),
            nn.Linear(ds_feature_size, ds_feature_size),
            nn.ReLU(),
            nn.Linear(ds_feature_size, vision_feature_size),
            nn.ReLU()
        )

        self.fcs = nn.Sequential(
            nn.Linear(vision_feature_size, vision_feature_size),
            nn.ReLU(),
            nn.Linear(vision_feature_size, vision_feature_size),
            nn.ReLU()
        )

        self.output = nn.Linear(vision_feature_size, out_sz)

        [kaiming_normal(layer.weight) for layer in list(self.dim_alignment.modules()) if
         type(layer) == nn.Linear]
        [kaiming_normal(layer.weight) for layer in list(self.fcs.modules()) if type(layer) == nn.Linear]
        kaiming_normal(self.output.weight)

    def forward(self, data):
        img, x_cat, x_cont = data
        x1 = self.vision_model(img)
        x2 = self.embedding_model.forward(x_cat, x_cont)
        x2 = self.dim_alignment(x2)
        x = x1 + x2
        x = self.fcs(x)
        return self.output(x)
