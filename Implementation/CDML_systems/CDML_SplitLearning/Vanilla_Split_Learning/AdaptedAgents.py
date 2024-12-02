from ...base import Base_Agents
from . import AdaptedRoles as role
from ...base import Base_Methods as meth
import torch
import torch.nn as nn


class Tra(Base_Agents.basic_Tra):
    def trainMLModel(self, epochs=None):
        role.trainer.trainMLModel(self, epochs=epochs)
    def awaitUpdate(self):
        role.trainer.awaitUpdate(self)


class CooSel(Base_Agents.basic_CooSel):
    pass

class TraUpd(role.updater, role.trainer, meth.applyForCoalition, meth.transmitInterimResult):
    def __init__(self, id, trainloader,communication_protocoll, ML_Model = None, ip=None, agent="TraUpd", server_address=None,criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None, ML_Model_update=None, update_optimizer =None):
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, agent=agent, server_address=server_address, ML_Model = ML_Model, criterion=criterion,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader, ML_Model_update = ML_Model_update, 
                         update_optimizer =update_optimizer)
    def initialization_phase(self, period=None, purposeOfSystem=None, agentRequirements=None):
        role.trainer.initialization_phase(self, period=period, purposeOfSystem=purposeOfSystem, agentRequirements=agentRequirements)
        role.updater.initialization_phase(self, period=period, purposeOfSystem=purposeOfSystem, agentRequirements=agentRequirements)


class ConTraUpd(role.configurator, role.trainer, role.updater):
    def __init__(self, id, trainloader,communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="ConTraUpd", server_address=None, coalition=None, MLTask=None, criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None, ML_Model_update=None, update_optimizer =None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask,
                         criterion=criterion, optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader, ML_Model_update = ML_Model_update,
                         update_optimizer =update_optimizer)
    def initialization_phase(self, period=None, purposeOfSystem=None, agentRequirements=None):
        return role.configurator.initialization_phase(self)
    

class ConCooSelUpd(role.configurator, role.coordinator, role.selector, role.updater):
    def __init__(self, id, communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="ConCooSelUpd", server_address=None, coalition=None, MLTask=None, ML_Model_update=None, device = None, update_optimizer =None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, 
                         MLTask=MLTask, ML_Model_update = ML_Model_update, device = device, update_optimizer =update_optimizer)
    def initialization_phase(self, period=None, purposeOfSystem=None, agentRequirements=None):
        return role.configurator.initialization_phase(self, period=period, purposeOfSystem=purposeOfSystem, agentRequirements=agentRequirements)
    def operation_phase(self, ready_agents=3, rounds=1):
        for i in range(0,rounds):
            print("Round: " + str(i))
            role.selector.operation_phase(self, ready_agents=ready_agents, rounds=1)
            role.updater.operation_phase(self, rounds=1)
    

class CooSelTraUpd(role.coordinator, role.selector, role.trainer, role.updater):
    def __init__(self, id, trainloader, communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="CooSelTraUpd", server_address=None, coalition=None, MLTask=None,criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None, ML_Model_update=None, update_optimizer =None):
         super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask, criterion=criterion,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader, ML_Model_update = ML_Model_update,update_optimizer =update_optimizer)


class ConCooSelTraUpd(role.configurator, role.coordinator, role.selector, role.trainer, role.updater):
    def __init__(self, id, trainloader, communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="ConCooSelTraUpd", server_address=None, coalition=None, MLTask=None, criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None, ML_Model_update=None, update_optimizer =None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask,
                         criterion=criterion, update_optimizer =update_optimizer,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader, ML_Model_update = ML_Model_update)
    
    def initialization_phase(self, period=None, purposeOfSystem=None, agentRequirements=None):
        return role.configurator.initialization_phase(self, period=period, purposeOfSystem=purposeOfSystem, agentRequirements=agentRequirements)
    
    def operation_phase(self, ready_agents=3, rounds=1):
        for i in range(0,rounds):
            print("Round: " + str(i))
            role.selector.operation_phase(self, ready_agents=ready_agents, rounds=1)
            role.trainer.operation_phase(self, rounds=1)
            role.updater.operation_phase(self, rounds=1)
            






