from . import Base_Methods as meth
from . import Impl_Methods
import torch
import torch.nn as nn
from . import Base_Constants

class basic_configurator(meth.awaitApplication, meth.applyForCoalition, meth.informApplicant, meth.provideMLTask, meth.assignSelectorAgent,
                         meth.registerCoalition, meth.defineInterimResults, meth.defineInitialMLModel, meth.decideOnApplication, meth.defineRepository):
    def __init__(self, id=-1, communication_protocoll=None, ip = None, repository=None, interimResultsDefinition=None, application_list=None, coalition_members=None, 
                 agent="Con", server_address=None, coalition=None, MLTask=None, **kwargs):
        self.coalition = coalition
        
        self.interimResultsDefinition = interimResultsDefinition
        self.updater_agents = []
        self.TraUpdPairing = []

        self.MLTask = MLTask
        if application_list is None:
            self.application_list = []
        else:
            self.application_list = application_list
        if coalition_members is None:
            self.coalition_members = []
        else:
            self.coalition_members = coalition_members
        super().__init__(**kwargs)
        self.id = id
        if ip is None:
            self.ip = 'localhost', 10000+int(self.id)
        else:
            self.ip = ip
        self.communication_protocoll = communication_protocoll
        self.repository = repository
        self.agent = Impl_Methods.AgentRoleConversion(agent)
        if server_address is not None:
            self.server_address = server_address
        else:
            self.server_address = 'localhost', 10000
        
        
    
    def initialization_phase(self, period = None, purposeOfSystem = None, agentRequirements = None):
        if self.server_address != self.ip:
            #we are not master so we don't handle the application process
            self.applyForCoalition()
            while True:
                application_response = self.communication_protocoll.receive_information(self)
                if application_response.tag == Base_Constants.possible_tags[8]:
                    break
            #Application response will be true or false depending on decideOnApplication
            self.chosen = application_response.object.result
            #when we are trainer or updater the coordinator is our selector
            #should we have been chosen, we will need to wait for the ML Task and if we are Tra or Upd, we need to wait for the paired trainer or updater
            if self.chosen:
                self.agent = application_response.object.agent.roles

                while True:
                    information = self.communication_protocoll.receive_information(self)
                    if(information.tag == Base_Constants.possible_tags[2]):
                        self.MLTask = information.object
                        break
                while True:
                    information = self.communication_protocoll.receive_information(self)
                    if information.tag==Base_Constants.possible_tags[1]:
                        break
        else:
            self.defineInitialMLModel(model= self.MLTask)
            self.defineInterimResults(interimResults=self.interimResultsDefinition)
            self.defineRepository()
            self.registerCoalition(purposeOfSystem, agentRequirements)

            while True:
                #We simulate our own application to us self
                if len(self.coalition_members) > 0:
                    result = self.awaitApplication(timeperiod=period)
                else:
                    application_object = Impl_Methods.CoalitionEntry(self.id, self.ip, self.agent)
                    application_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[6], object=application_object)
                    if period[Base_Constants.period["signal"]] == "time":
                        result = application_message, period[Base_Constants.period["signal key value"]]
                    else:
                        result = application_message, None
                    
                
                if result is None:
                    #we reached our target
                    break
                else:
                    applicant, remaining_time = result
                    if(remaining_time is not None):
                        #Here a time period has been specified, else we use an amount
                        period = ("Initialization", "time", remaining_time)
                    accepted, rejected = self.decideOnApplication(applicant.object)
                    self.informApplicant(accepted, rejected)
                    self.application_list.append(applicant.object)                            
                    if len(accepted)!= 0:
                        self.provideMLTask(applicant.object)
                    for appl in accepted: 
                        if len(self.updater_agents) == 0 and "updater" in appl.agent.roles:
                            self.updater_agents.append(appl)
                            for agent in self.coalition_members:
                                if "trainer" in agent.agent.roles:
                                    self.assignInterimResultRecipient(agent)
                        elif "updater" in appl.agent.roles:
                            self.updater_agents.append(appl)
                        if "trainer" in appl.agent.roles and len(self.updater_agents) != 0:
                            self.assignInterimResultRecipient(appl)

                        self.coalition_members.append(appl)
            

            self.assignSelectorAgent()

            for coalition_member in self.coalition_members:
                if coalition_member.id != self.id:
                    message = Impl_Methods.Message(tag = Base_Constants.possible_tags[1], object = "")
                    self.communication_protocoll.send_information(self, message=message, communication_address=coalition_member.ip)


class basic_coordinator(meth.awaitApplication, meth.applyForCoalition, meth.informApplicant, meth.provideMLTask,
                         meth.assignInterimResultRecipient, meth.decideOnApplication):
    def __init__(self, id=-1, communication_protocoll=None, ip = None, repository=None, interimResultsDefinition=None, application_list=None, coalition_members=None, 
                 agent="Con", server_address=None, coalition=None, MLTask=None, **kwargs):
        self.coalition = coalition
        
        self.interimResultsDefinition = interimResultsDefinition
        self.updater_agents = []
        self.TraUpdPairing = []

        self.MLTask = MLTask
        if application_list is None:
            self.application_list = []
        else:
            self.application_list = application_list
        if coalition_members is None:
            self.coalition_members = []
        else:
            self.coalition_members = coalition_members
        super().__init__(**kwargs)
        self.id = id
        if ip is None:
            self.ip = 'localhost', 10000+int(self.id)
        else:
            self.ip = ip
        self.communication_protocoll = communication_protocoll
        self.repository = repository
        self.agent = Impl_Methods.AgentRoleConversion(agent)
        if server_address is not None:
            self.server_address = server_address
        else:
            self.server_address = 'localhost', 10000

class basic_selector(meth.applyForCoalition, meth.selectAgent, meth.announceAgentSelection, meth.awaitReadiness):
    def __init__(self, id=-1, communication_protocoll=None, ip = None, server_address = None, agent = "Sel", **kwargs):
        super().__init__(**kwargs)
        self.id = id
        if ip is None:
            self.ip = 'localhost', 10000+int(self.id)
        else:
            self.ip = ip
        self.communication_protocoll = communication_protocoll
        self.repository = None
        self.readyAgents = []
        self.agent = Impl_Methods.AgentRoleConversion(agent)
        if server_address is not None:
            self.server_address = server_address
        else:
            self.server_address = 'localhost', 10000
    
    def operation_phase(self, ready_agents = 1, rounds=1):
        self.signalReadiness()
        self.awaitReadiness(amount = ready_agents)
        self.selectAgent()
        self.announceAgentSelection()
        self.readyAgents=[]


class basic_trainer(meth. awaitSelectionSignal, meth.trainMLModel,  meth.signalReadiness, meth.awaitUpdate, meth.testMLModel):
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
            self.optimizer = torch.optim.SGD
        else:
            self.optimizer = optimizer
        self.model_optimizer = self.optimizer(self.ML_Model.parameters(), lr=0.001, momentum=0.9)
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
        
        

    def initialization_phase(self, period = None, purposeOfSystem = None, agentRequirements = None):
        #we are not master so we don't handle the application process
        self.applyForCoalition()
        response = False
        while not response:
            application_response = self.communication_protocoll.receive_information(self)
            if application_response.tag == Base_Constants.possible_tags[8]:
                response=True
        #Application response will be true or false depending on decideOnApplication
        self.chosen = application_response.object.result
        #when we are trainer or updater the coordinator is our selector
        #should we have been chosen, we will need to wait for the ML Task and if we are Tra or Upd, we need to wait for the paired trainer or updater
        if self.chosen:
            ML_Task = True
            updating_agent = True
            
            while ML_Task or updating_agent or self.selector_agent:
                information = self.communication_protocoll.receive_information(self)
                if(information.tag == Base_Constants.possible_tags[2] and ML_Task):
                    self.MLTask = information.object
                    ML_Task = False
                elif(information.tag == Base_Constants.possible_tags[4] and updating_agent):       
                    self.updater = information.object
                    updating_agent=False
                elif (information.tag == Base_Constants.possible_tags[13] and self.selector_agent):
                        self.selector = information.object
                        self.selector_agent = False
            while True:
                information = self.communication_protocoll.receive_information(self)
                if information.tag==Base_Constants.possible_tags[1]:
                    break


    def operation_phase(self, rounds=1):
        for i in range(0,rounds):
            print("Training round: " + str(i))
            if self.actions["trainer"]==False:
                self.signalReadiness()
                self.awaitSelectionSignal()
            self.model_optimizer = self.optimizer(self.ML_Model.parameters(), lr=0.001, momentum=0.9)
            self.trainMLModel(epochs=2)
            self.transmitInterimResult()
            self.awaitUpdate()
            self.testMLModel()
            self.actions["trainer"]=False
         


class basic_updater(meth.awaitSelectionSignal, meth.transmitInterimResult, meth.signalReadiness, meth.updateMLModel, meth.awaitInterimResults):
    #needs transmitInterimResults and applyForCoalition
    def __init__(self, id=-1, communication_protocoll=None,ip=None, agent = "Upd", server_address=None, **kwargs):
        self.chosen = False
        self.selector = None
        self.trainers = []
        self.actions = {"trainer":False, "updater": False}
        self.agent_ready_to_update = []
        self.receivedInterimResults = []

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


    def initialization_phase(self, period = None, purposeOfSystem = None, agentRequirements = None):
        #we are not master so we don't handle the application process
        self.applyForCoalition()
        response = False
        while not response:
            application_response = self.communication_protocoll.receive_information(self)
            if application_response.tag == Base_Constants.possible_tags[8]:
                response=True
        #Application response will be true or false depending on decideOnApplication
        self.chosen = application_response.object.result
        #when we are trainer or updater the coordinator is our selector
        #should we have been chosen, we will need to wait for the ML Task and if we are Tra or Upd, we need to wait for the paired trainer or updater
        if self.chosen:
            self.agent = application_response.object.agent
            while True:
                information = self.communication_protocoll.receive_information(self)

                if information.tag==Base_Constants.possible_tags[1]:
                    break
                elif information.tag==Base_Constants.possible_tags[2]:
                    self.MLTask = information.object
                elif information.tag==Base_Constants.possible_tags[3]:
                    self.trainers.append(information.object)
                elif (information.tag == Base_Constants.possible_tags[13]):
                    self.selector = information.object
    

    def operation_phase(self, rounds=1):
        for i in range(0,rounds):
            if self.actions["updater"]==False:
                self.signalReadiness()
                self.awaitSelectionSignal()
            self.awaitInterimResults()
            self.updateMLModel()
            self.actions["updater"]=False
            self.agent_ready_to_update = []