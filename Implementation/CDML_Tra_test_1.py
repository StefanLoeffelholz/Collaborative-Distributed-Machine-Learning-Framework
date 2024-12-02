import torch
from CDML_systems.CDML_FederatedLearning.FedProx.AdaptedAgents import Tra as Federated_Learning_Tra
#from CDML_systems.CDML_AssistedLearning.GAL.AdaptedAgents import Tra as Assisted_Learning_Tra
#from CDML_systems.CDML_SplitLearning.Vanilla_Split_Learning.AdaptedAgents import Tra as Split_Learning_Tra

from CDML_systems.base import Impl_Methods
from CDML_systems.base import Base_Dataloader
from CDML_systems.base import Base_Model

from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms


import random
import numpy as np

def set_seed(seed):
    random.seed(seed)  # Python's random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch (GPU)
        torch.cuda.manual_seed_all(seed)  # All GPU devices
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable cuDNN optimization for reproducibility

trainset = CIFAR100("./data_cifar100", train=True, download=True)

def generate_dataloader_one_third_of_data():
    trainset = CIFAR100("./data_cifar100", train=True, download=True)
    shuffler = Base_Dataloader.sort_shuffle_data()
    partition10_1 = [1,1,1,1,1,1,1,1,1,1]
    partition10_0 = [0,0,0,0,0,0,0,0,0,0]

    partition100_341_660 = []
    for i in range(0,3): partition100_341_660.extend(partition10_1)
    partition100_341_660.extend([1,1,1,1,0,0,0,0,0,0])
    for i in range(0,6): partition100_341_660.extend(partition10_0)

    partition_340_331_330 = []
    for i in range(0,3): partition_340_331_330.extend(partition10_0)
    partition_340_331_330.extend([0,0,0,0,1,1,1,1,1,1])
    for i in range(0,2): partition_340_331_330.extend(partition10_1)
    partition_340_331_330.extend([1,1,1,1,1,1,1,0,0,0])
    for i in range(0,3): partition_340_331_330.extend(partition10_0)

    partition_670_331 = []
    for i in range(0,6): partition_670_331.extend(partition10_0)
    partition_670_331.extend([0,0,0,0,0,0,0,1,1,1])
    for i in range(0,3): partition_670_331.extend(partition10_1)

    partitions = [partition100_341_660, partition_340_331_330, partition_670_331]
    shuffled_trainsets, shuffled_labels = shuffler.partition_data(trainset.data, trainset.targets,partition_amount=3, partitioning=partitions)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = Base_Dataloader.CustomImageDataset(labels=shuffled_labels[1], data=shuffled_trainsets[1],transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0,shuffle=True)
    return trainloader


trainloader = generate_dataloader_one_third_of_data()


model = Base_Model.resnet18()

"""model, server_model = Base_Model.split_resnet18()
model.load_state_dict(torch.load("./param_split_client", weights_only=True))"""

model.load_state_dict(torch.load("./param", weights_only=True))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tra = Federated_Learning_Tra(1, trainloader=trainloader,communication_protocoll=Impl_Methods.Communication, device=DEVICE, ML_Model=model)

tra.initialization_phase()
tra.operation_phase(rounds=20)