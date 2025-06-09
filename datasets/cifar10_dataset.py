from datasets.base_dataset import BaseDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

class Cifar10Dataset(BaseDataset):
    """This class implements the CIFAR-10 dataset."""

    def __init__(self, is_train):
        """Initialize the class; this function is called by the subclass's __init__ function."""
        super().__init__(is_train)
        # Set the training flag
        self.is_train = is_train
        # Define the transformations for the dataset
        self.transform, self.inv_transform = self.get_transform()
        # Load the CIFAR-10 dataset
        self.dataset = CIFAR10(root='./data/cifar10', train=is_train, download=True, transform=self.transform)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        image, label = self.dataset[index]
        return {'image': image, 'label': label}

    def get_transform(self):
        """Return a list of transformations to be applied to the dataset."""
        if self.is_train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        inv_transform = transforms.Compose([
            transforms.Normalize((-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010),
                                 (1 / 0.2023, 1 / 0.1994, 1 / 0.2010))
        ])
        
        return transform, inv_transform