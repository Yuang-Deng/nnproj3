import torch.nn as nn
from functions import ReverseLayerF


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_fc1', nn.Linear(310, 128))
        self.feature.add_module('f_bn1', nn.BatchNorm1d(128))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_drop1', nn.Dropout())
        self.feature.add_module('f_conv2', nn.Linear(128, 128))
        self.feature.add_module('f_bn2', nn.BatchNorm1d(128))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_drop2', nn.Dropout())

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(128, 64))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(64, 64))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(64, 3))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(128, 64))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(64))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_drop1', nn.Dropout())
        self.domain_classifier.add_module('d_fc2', nn.Linear(64, 64))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(64))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(64, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.float()
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
