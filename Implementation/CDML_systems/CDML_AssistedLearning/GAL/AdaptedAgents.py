import time
import torch
import torch.nn as nn
from ...base import Base_Agents
from ...base import Impl_Methods
from ...base import Base_Constants
from . import AdaptedRoles as role


class Tra(Base_Agents.basic_Tra):
    def __init__(self, id, trainloader,communication_protocoll, ML_Model = None, ip=None, agent="Tra", server_address=None,criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, agent=agent, server_address=server_address, ML_Model = ML_Model, criterion=criterion,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader)
        self.pseudo_residuals = None
    def trainMLModel(self, epochs=None):
        role.trainer.trainMLModel(self, epochs=epochs)
    def awaitUpdate(self):
        role.trainer.awaitUpdate(self)
    def operation_phase(self, rounds):
        role.trainer.operation_phase(self, rounds=rounds)

class CooSel(Base_Agents.basic_CooSel):
    pass

class TraUpd(Base_Agents.basic_TraUpd):
    def __init__(self, id, trainloader,communication_protocoll, ML_Model = None, ip=None, agent="TraUpd", server_address=None,criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None):
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, agent=agent, server_address=server_address, ML_Model = ML_Model, criterion=criterion,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader)
    def trainMLModel(self, epochs=None):
        role.trainer.trainMLModel(self, epochs=epochs)
    def awaitUpdate(self):
        role.trainer.awaitUpdate(self)
    def updateMLModel(self):
        role.updater.updateMLModel(self)
    def operation_phase(self, rounds):
        role.trainer.operation_phase(self, rounds=rounds)

class ConTraUpd(Base_Agents.basic_ConTraUpd):
    def __init__(self, id, trainloader,communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="ConTraUpd", server_address=None, coalition=None, MLTask=None, criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask,
                         criterion=criterion, optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader)
    def trainMLModel(self, epochs=None):
        role.trainer.trainMLModel(self, epochs=epochs)
    def awaitUpdate(self):
        role.trainer.awaitUpdate(self)
    def updateMLModel(self):
        role.updater.updateMLModel(self)
    def operation_phase(self, rounds):
        role.trainer.operation_phase(self, rounds=rounds)

class ConCooSelUpd(Base_Agents.basic_ConCooSelUpd):
    def __init__(self, id,communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, application_list=None, coalition_members=None, 
                 agent="CooSelTraUpd", server_address=None, coalition=None, MLTask=None, ML_Model=None, device = None, optimizer = None, criterion = nn.CrossEntropyLoss(),
                 trainloader=None):
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask)
        self.device = device
        self.ML_Model = ML_Model
        
        self.trainloader = trainloader
        self.criterion = criterion
        if optimizer is None:
            self.optimizer = torch.optim.SGD
        else:
            self.optimizer = optimizer
        self.model_optimizer = self.optimizer(self.ML_Model.parameters(), lr=0.01, momentum=0.9)


    def updateMLModel(self):
        role.updater.updateMLModel(self)
    def operation_phase(self, ready_agents=3, rounds=1):
        start_time = time.time()
        for i in range(0,rounds):
            role.selector.operation_phase(self, ready_agents=ready_agents, rounds=1)
            role.updater.operation_phase(self, rounds=1)
            for participant in self.coalition_members:
                if self.id != participant.id:
                    next_message = Impl_Methods.Message(tag = Base_Constants.possible_tags[12], object="")
                    self.communication_protocoll.send_information(self, next_message, communication_address=participant.ip) 


class CooSelTraUpd(Base_Agents.basic_CooSelTraUpd):
    def __init__(self, id, trainloader, communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="CooSelTraUpd", server_address=None, coalition=None, MLTask=None,criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None):
         super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask, criterion=criterion,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader)
    def trainMLModel(self, epochs=None):
        role.trainer.trainMLModel(self, epochs=epochs)
    def awaitUpdate(self):
        role.trainer.awaitUpdate(self)
    def updateMLModel(self):
        role.updater.updateMLModel(self)
    def operation_phase(self, rounds):
        role.trainer.operation_phase(self, rounds=rounds)

class ConCooSelTraUpd(Base_Agents.basic_ConCooSelTraUpd):
    def __init__(self, id, trainloader, communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="ConCooSelTraUpd", server_address=None, coalition=None, MLTask=None, criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask,
                         criterion=criterion,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader)
    def trainMLModel(self, epochs=None):
        role.trainer.trainMLModel(self, epochs=epochs)
    def awaitUpdate(self):
        role.trainer.awaitUpdate(self)
    def updateMLModel(self):
        role.updater.updateMLModel(self)
    def operation_phase(self, ready_agents=3, rounds=1):
        for i in range(0,rounds):
            role.selector.operation_phase(self, ready_agents=ready_agents, rounds=1)
            if self.actions["updater"]==False:
                self.signalReadiness()
                self.awaitSelectionSignal()
            if self.actions["trainer"]==False:
                self.signalReadiness()
                self.awaitSelectionSignal()
            self.updateMLModel()
            self.awaitUpdate()
            self.trainMLModel(epochs=2)
            self.transmitInterimResult()
            self.testMLModel()
            self.actions["trainer"]=False
            self.awaitInterimResults()
            self.actions["updater"]=False
            self.agent_ready_to_update = []





