import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torchvision import datasets, transforms

# torchvision 默认会依次尝试官方源和 ossci 备用源，ossci 国内可直连
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

train_set = datasets.MNIST(root='./data', train=True, download=True,
                           transform=transforms.ToTensor())
test_set = datasets.MNIST(root='./data', train=False, download=True,
                          transform=transforms.ToTensor())