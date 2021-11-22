from collections import OrderedDict
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, drop=0.5, num_classes=10):
        super(SmallCNN, self).__init__()

        self.num_channels = 1
        self.num_classes=num_classes

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 256)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(256, 256)),
            ('relu2', activ),
            ('fc3', nn.Linear(256, self.num_classes)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def intermediate_size(self):
        return [64 * 4 * 4, self.num_classes]

    def intermediate_forward(self, input):
        return [self.layer_1(input), self.forward(input)]

    def predict(self, x):
        import numpy as np
        import torch
        batch_size = 32
        preds = np.zeros((x.shape[0], self.num_classes))
        for i in range(int(x.shape[0] / batch_size)):
            preds[i * batch_size:(i + 1) * batch_size] = self.forward(
                torch.tensor(x[i * batch_size:(i + 1) * batch_size]).float().cuda()).cpu().detach().numpy()
        return torch.from_numpy(preds)

    def layer_1(self, input):
        return self.feature_extractor(input).contiguous().view(-1, 64 * 4 * 4)

    def forward(self, input):
        features = self.feature_extractor(input)
        #print(features.size())
        logits = self.classifier(features.contiguous().view(-1, 64 * 4 * 4))
        return logits
