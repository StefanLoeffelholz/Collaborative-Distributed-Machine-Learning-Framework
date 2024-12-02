from . import Base_Roles as role
from . import Base_Methods as meth
import torch
import torch.nn as nn


class basic_Tra(role.basic_trainer, meth.applyForCoalition, meth.transmitInterimResult):
    def __init__(self, id, trainloader,communication_protocoll, ML_Model = None, ip=None, agent="Tra", server_address=None,criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, agent=agent, server_address=server_address, ML_Model = ML_Model, criterion=criterion,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader)


class basic_CooSel(role.basic_coordinator, role.basic_selector):
    def __init__(self, id, communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="CooSel", server_address=None, coalition=None, MLTask=None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask)


class basic_TraUpd(role.basic_updater, role.basic_trainer, meth.applyForCoalition, meth.transmitInterimResult):
    def __init__(self, id, trainloader,communication_protocoll, ML_Model = None, ip=None, agent="TraUpd", server_address=None,criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None):
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, agent=agent, server_address=server_address, ML_Model = ML_Model, criterion=criterion,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader)
    def initialization_phase(self, period=None, purposeOfSystem=None, agentRequirements=None):
        role.basic_trainer.initialization_phase(self, period=period, purposeOfSystem=purposeOfSystem, agentRequirements=agentRequirements)
        role.basic_updater.initialization_phase(self, period=period, purposeOfSystem=purposeOfSystem, agentRequirements=agentRequirements)


class basic_ConTraUpd(role.basic_configurator, role.basic_trainer, role.basic_updater):
    def __init__(self, id, trainloader,communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="ConTraUpd", server_address=None, coalition=None, MLTask=None, criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask,
                         criterion=criterion, optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader)
    def initialization_phase(self, period=None, purposeOfSystem=None, agentRequirements=None):
        return role.basic_configurator.initialization_phase(self)
    

class basic_ConCooSelUpd(role.basic_configurator, role.basic_coordinator, role.basic_selector, role.basic_updater):
    def __init__(self, id, communication_protocoll, ip = None, repository=None, interimResultsDefinition=None,application_list=None, coalition_members=None, 
                 agent="ConCooSelUpd", server_address=None, coalition=None, MLTask=None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, 
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask)
    def initialization_phase(self, period=None, purposeOfSystem=None, agentRequirements=None):
        return role.basic_configurator.initialization_phase(self, period=period, purposeOfSystem=purposeOfSystem, agentRequirements=agentRequirements)
    def operation_phase(self, ready_agents=3, rounds=1):
        for i in range(0,rounds):
            print("Round: " + str(i))
            role.basic_selector.operation_phase(self, ready_agents=ready_agents, rounds=1)
            role.basic_updater.operation_phase(self, rounds=1)

class basic_CooSelTraUpd(role.basic_coordinator, role.basic_selector, role.basic_trainer, role.basic_updater):
    def __init__(self, id, trainloader, communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="CooSelTraUpd", server_address=None, coalition=None, MLTask=None,criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None):
         super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask, criterion=criterion,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader)


class basic_ConCooSelTraUpd(role.basic_configurator, role.basic_coordinator, role.basic_selector, role.basic_trainer, role.basic_updater):
    def __init__(self, id, trainloader, communication_protocoll, ip = None, repository=None, interimResultsDefinition=None, ML_Model=None, application_list=None, coalition_members=None, 
                 agent="ConCooSelTraUpd", server_address=None, coalition=None, MLTask=None, criterion=nn.CrossEntropyLoss(),
                 optimizer = torch.optim.SGD, device = None, selector_agent=None):
        # Pass id, optimizer, and data up the chain
        super().__init__(id=id, communication_protocoll=communication_protocoll, ip = ip, repository=repository, interimResultsDefinition=interimResultsDefinition, ML_Model=ML_Model,
                         application_list=application_list, coalition_members=coalition_members, agent=agent, server_address=server_address, coalition = coalition, MLTask=MLTask,
                         criterion=criterion,
                         optimizer=optimizer, device=device, selector_agent=selector_agent, trainloader=trainloader)
        
        self.results = []
        
    
    def initialization_phase(self, period=None, purposeOfSystem=None, agentRequirements=None):
        return role.basic_configurator.initialization_phase(self, period=period, purposeOfSystem=purposeOfSystem, agentRequirements=agentRequirements)
    
    def operation_phase(self, ready_agents=3, rounds=1):
        for i in range(0,rounds):
            print("Round: " + str(i))
            role.basic_selector.operation_phase(self, ready_agents=ready_agents, rounds=1)
            role.basic_trainer.operation_phase(self, rounds=1)
            role.basic_updater.operation_phase(self, rounds=1)
            