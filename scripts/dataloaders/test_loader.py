from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import random

np.random.seed(42)
random_colors = np.random.randint(255,size=(10,3))
digits_mean_color = np.arange(0,10)
np.random.shuffle(digits_mean_color)


class TestDataset(Dataset):
    def __init__(self, training_size, data, random_colors=random_colors, mean_colors = digits_mean_color):
        self.data = data
        self.size = training_size
        self.random_colors = random_colors
        self.mean_colors = mean_colors
        
    def __len__(self):
        return self.data.data.shape[0] - self.size

    def __getitem__(self, index):

        index = index + self.size
        
        gray_image = self.data.data[index].float()/255.0
        label = self.data.targets[index] 
        color_label = random.randint(0,9)
        color= np.random.normal(self.random_colors[color_label],30,size=(3))
        color = np.clip(color,0,255)
        color_red = int(color[0] / 32)
        color_green = int(color[1] /32)
        color_blue = int(color[2] /32)
        colored_image = (gray_image.view(1,28,28)*(torch.from_numpy(color).float().view(3,1,1))/255.0)
        return (colored_image,label,color_red,color_green,color_blue)



def get_test_dataloader():
    data = datasets.mnist.MNIST("./", download=False)
    training_size = int(0.9*data.data.shape[0])
    test_dataset = TestDataset(training_size=training_size, data = data)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=2)
    return test_dataloader
