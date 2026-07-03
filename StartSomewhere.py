#testing if pytorch worked
import torch
import numpy as np

print("PyTorch version:", torch.__version__)
print("Is CUDA (CPU) available?", torch.cuda.is_available())

#yay now that pytorch works we can start our project!

#will be using https://docs.pytorch.org/tutorials/beginner/basics/intro.html

# I will then check out https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch and https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/

#a tensor is basically an array and you can dedcide it's shape directly
#numpy arrays can be directly converted to tensors and vice versa using the .from_numpy() and .numpy() methods

#Tensors have attributes such as shape, dtype, device
tensor = torch.rand(3,4) #tensor with 3 rows and 4 colums with random float values

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#there are lots of operations that can be done to tensors, over a thousand!
#you can copy the data of tensors from other devices but it may strain the device
#to make a model faster, we will want to send the data to the GPU as the accelerator like  this:
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

#here are some list operation done on a tensor:
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

#look up more awesome tensor operations here: https://pytorch.org/docs/stable/tensors.html

#tensors and numpy arrays can share the same memory when you convert them
#going from tensor to numpy:
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

#going from numpy to tensor:
n = np.ones(5)
t = torch.from_numpy(n)

#******************************************************************
#loading a dataset and Dataloaders

#"datasets and dataloaders are used to load data into the model for training and testing"
#-chud gpt

#just added this to my GitHub so you KNOW it's serious now 7/2/26

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
#I should have just imported all the stuff I need to use a dataset
#lets make the training and test data for the MNIST dataset

training_data = datasets.MNIST(
    root="data",
    train = True,
    download = True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
)
#sweet, MNIST from datasets has been initialized to root as data, marked as training, downloaded, and transformed to usable data I assume

test_data = datasets.MNIST(
    root = "data",
    train = False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
)

#okay now the test and training data is loaded into the program

#uh how the heck do I visualize this dataset?



