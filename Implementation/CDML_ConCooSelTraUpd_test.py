import torch
from CDML_systems.CDML_FederatedLearning.FedProx.AdaptedAgents import ConCooSelTraUpd as Federated_Learning_ConCooSelTraUpd
#from CDML_systems.CDML_AssistedLearning.GAL.AdaptedAgents import Assisted_Learning_ConCooSelUpd
#from CDML_systems.CDML_SplitLearning.Vanilla_Split_Learning.AdaptedAgents import ConCooSelTraUpd as Split_Learning_ConCooSelTraUpd
from CDML_systems.base import Base_Constants
from CDML_systems.base import Base_Model
from CDML_systems.base import Impl_Methods
from CDML_systems.base import Base_Dataloader
import torch.nn as nn

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

# Set the seed
seed = 42
set_seed(seed)

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
    dataset = Base_Dataloader.CustomImageDataset(labels=shuffled_labels[0], data=shuffled_trainsets[0],transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0,shuffle=True)
    return trainloader


trainloader = generate_dataloader_one_third_of_data()


model = Base_Model.resnet18()
model.load_state_dict(torch.load("./param", weights_only=True))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
concooseltraupd = Federated_Learning_ConCooSelTraUpd(0, trainloader=trainloader,
                                  communication_protocoll=Impl_Methods.Communication, MLTask=model
                                  , agent="ConCooSelTraUpd", ML_Model=model,interimResultsDefinition=Base_Constants.possible_results["Federated"], 
                                  device=DEVICE )

"""concooseltraupd = Assisted_Learning_ConCooSelUpd(0,communication_protocoll=Impl_Methods.Communication, MLTask=model,
                               agent="ConCooSelUpd", interimResultsDefinition=Base_Constants.possible_results["Federated"], trainloader=trainloader, device=DEVICE,
                                 ML_Model=model)"""

"""
model, server_model = Base_Model.split_resnet18()
model.load_state_dict(torch.load("./param_split_client", weights_only=True))
server_model.load_state_dict(torch.load("./param_split_server_model", weights_only=True))
concooseltraupd = Split_Learning_ConCooSelTraUpd(0, trainloader=trainloader,
                                  communication_protocoll=Impl_Methods.Communication, MLTask=model
                                  , agent="ConCooSelTraUpd", ML_Model=model,interimResultsDefinition=Base_Constants.possible_results["Federated"], 
                                  device=DEVICE , ML_Model_update=server_model)"""

period = ("Initialization", "amount", 3)
concooseltraupd.initialization_phase(period=period)
concooseltraupd.operation_phase(rounds=20)
model = concooseltraupd.ML_Model

