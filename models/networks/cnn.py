import torch.nn as nn


class CNN(nn.Module):
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
        super(CNN, self).__init__()
        self.input_nc = input_nc
        self.num_classes = num_classes
        self.net_name = net_name
        self.use_dropout = use_dropout
        self.pool_type = pool_type

        # Feature extraction part using Sequential
        self.features = nn.Sequential(
            nn.Conv2d(input_nc, 32, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) if pool_type == 'max' else nn.AvgPool2d(2),
            nn.Dropout(0.25) if use_dropout else nn.Identity(),
        )

        # Classification part using Sequential
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),  # You may need to adjust this size based on your input dimensions
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.Linear(128, num_classes)
        )

        # Final activation
        self.output_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.output_activation(x)
        return x