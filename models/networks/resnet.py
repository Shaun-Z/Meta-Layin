import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    """A simple CNN classifier.
    """
    def __init__(self, input_nc, num_classes, net_name=None, use_dropout=False, pool_type='max'):
        """Initialize the CNN classifier.

        Parameters:
            input_nc (int) -- the number of channels in input images
            num_classes (int) -- the number of classes in the classification task
            ngf (int) -- the number of filters in the last conv layer
            net_name (str) -- name of the network architecture
            use_dropout (bool) -- if use dropout layers.
            pool_type (str) -- the type of pooling layer: max | avg
        """
        super(ResNet, self).__init__()
        if net_name == 'resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif net_name == 'resnet34':
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif net_name == 'resnet50':
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif net_name == 'resnet101':
            net = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif net_name == 'resnet152':
            net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            raise NotImplementedError(f'Classifier model name \033[92m[net_name]\033[0m is not recognized')
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    