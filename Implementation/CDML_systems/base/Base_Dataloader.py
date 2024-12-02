import time
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms as transforms
from torchvision.io import read_image
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import random
import os
import matplotlib.pyplot as plt

class sort_shuffle_data():
    def __init__(self):
        pass
        
    def get_class_i(self, data, data_labels, label):
        """
        x: trainset.train_data or testset.test_data
        y: trainset.train_labels or testset.test_labels
        i: class label, a number between 0 and len(self.classDict)
        return: x_i which are all the data entries of class i
        """
        # Convert to a numpy array
        data_labels = np.array(data_labels)
        # Locate position of labels that equal to i
        pos_i = np.argwhere(data_labels == label)
        # Convert the result into a 1-D list
        pos_i = list(pos_i[:, 0])
        # Collect all data that match the desired label
        x_i = [data[j] for j in pos_i]
        return x_i
    
    def get_labels(self, data_labels):
        different_labels = []
        for a in data_labels: 
            if len(different_labels) == 0:
                different_labels.append(a)
            elif a not in different_labels: 
                different_labels.append(a)
        return different_labels
    
    def partition_data(self, data, data_labels, class_labels=None, partitioning=None, partition_amount = 1, shuffling_function = random.Random(4).shuffle):
        partitioned_data = [[]for i in range(partition_amount)]
        partitioned_labels = [[]for i in range(partition_amount)]

        if class_labels is None:
            class_labels = self.get_labels(data_labels)
            class_labels.sort()
        
        seperated_classes = []
        for label in class_labels:
            seperated_classes.append(self.get_class_i(data, data_labels, label))

        if partitioning is None:
            partitioning = [[1 for j in range(len(class_labels))] for i in range(partition_amount)]
        
        total_partitionings = [0 for j in range(len(class_labels))]
        for partition in partitioning:
            for i in range(len(partition)):
                total_partitionings[i] += partition[i]
    
        for i in range(len(seperated_classes)):
            start_index = 0
            if total_partitionings[i]!=0:
                class_division_amount = len(seperated_classes[i]) // total_partitionings[i]
            else:
                class_division_amount = 0
            for k in range(len(partitioning)):
                partitioned_data[k].extend(seperated_classes[i][start_index:start_index+class_division_amount * partitioning[k][i]])
                partitioned_labels[k].extend(class_labels[i] for a in range(class_division_amount * partitioning[k][i]))
                start_index+=class_division_amount * partitioning[k][i]
        
        shuffled_seperated_data = []
        shuffled_seperated_labels = []
        for data, label in zip(partitioned_data, partitioned_labels):
            #image = data[0]
            #image = np.array(image)
            #print("Shape of image", image.shape)
            #image = np.transpose(image, (1, 2, 0)) 
            #plt.imshow(image)
            #plt.axis('off')  # Hide axes for cleaner display
            #plt.show()
            #print("Label:",label[0])
            data_and_label = list(zip(data, label))
            shuffling_function(data_and_label)
            new_data, new_labels = zip(*data_and_label) 
            shuffled_seperated_data.append(new_data)   
            shuffled_seperated_labels.append(new_labels)
        return shuffled_seperated_data, shuffled_seperated_labels

                
class CustomImageDataset(Dataset):
    def __init__(self, labels, data, transform=None, target_transform=None):
        self.img_labels = labels
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label