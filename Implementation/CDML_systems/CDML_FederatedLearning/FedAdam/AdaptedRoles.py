import torch
import torch.nn as nn
from ...base import Base_Roles
from ...base import Impl_Methods

class configurator(Base_Roles.basic_configurator):
    pass

class coordinator(Base_Roles.basic_coordinator):
    pass

class selector(Base_Roles.basic_selector):
    pass


class trainer(Base_Roles.basic_trainer):
    def __init__(self, id=-1, trainloader=None,communication_protocoll=None, ML_Model = None, ip=None, agent="Tra", server_address=None,criterion=nn.CrossEntropyLoss(),
                 optimizer = None, device = None, selector_agent=None, **kwargs):
        self.chosen = False
        self.ML_Model = ML_Model
        
        self.trainloader = trainloader
        self.criterion = criterion

        self.updater = None
        self.selector = None
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.ML_Model.to(self.device)
        if optimizer is None:
            self.optimizer = torch.optim.Adam
        else:
            self.optimizer = torch.optim.Adam
        self.model_optimizer = self.optimizer(self.ML_Model.parameters(), lr=0.1, betas=(0.9, 0.999))
        self.actions = {"trainer":False, "updater": False}
        self.interimResult = None
        self.training_loss = []
        if selector_agent is None:
            self.selector_agent = True
        else:
            self.selector_agent = False

        super().__init__(**kwargs)
        self.id = id
        if ip is None:
            self.ip = 'localhost', 10000+int(self.id)
        else:
            self.ip = ip
        self.communication_protocoll = communication_protocoll
        self.repository = None
        self.agent = Impl_Methods.AgentRoleConversion(agent)
        if server_address is not None:
            self.server_address = server_address
        else:
            self.server_address = 'localhost', 10000


    def operation_phase(self, rounds=1):
        for i in range(0,rounds):
            if self.actions["trainer"]==False:
                self.signalReadiness()
                self.awaitSelectionSignal()
            self.model_optimizer = self.optimizer(self.ML_Model.parameters(), lr=0.01, betas=(0.9, 0.999))
            self.trainMLModel(epochs=2)
            self.transmitInterimResult()
            self.awaitUpdate()
            self.testMLModel()
            self.actions["trainer"]=False


class updater(Base_Roles.basic_updater):
    pass






