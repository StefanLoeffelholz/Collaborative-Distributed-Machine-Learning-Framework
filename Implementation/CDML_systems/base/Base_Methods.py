import copy
from . import Impl_Methods
from . import Base_Constants
from . import Base_Model
import time
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from CDML_systems.base import Base_Model


class awaitApplication():
    def awaitApplication(self, timeperiod):
        """
        @param period defines the timeframe during which applications are received
        This method allows the configurator agents to wait for a specific time or until a certain amount of applicants is achieved.
        """
        if timeperiod is not None and timeperiod[Base_Constants.period["phase"]]=="Initialization":
            if timeperiod[Base_Constants.period["signal"]]=="time":
                details = timeperiod[Base_Constants.period["signal key value"]]

                if details is not None:
                    start_time = time.time()
                else:
                    print("No valid time, please try again!")
                    return None
                while True:
                    # Wait for a connection
                    
                    if(details is not None):
                        if((time.time()-start_time) > details):
                            print("Timer done!")
                            return None


                    applicant = self.communication_protocoll.receive_information(self)
                    if(applicant.tag ==Base_Constants.possible_tags[5]): #requesting specs from coalition
                        coalition_info_message = Impl_Methods.Message(tag = Base_Constants.possible_tags[7], object=self.coalition)
                        self.communication_protocoll.send_information(self, coalition_info_message, communication_address=applicant.ip)
                    elif applicant.tag == Base_Constants.possible_tags[6]: #Tag needs to be apply
                        if(applicant.object in self.application_list):
                            pass
                        else:
                            return applicant, time.time()-start_time
                        
            elif timeperiod[Base_Constants.period["signal"]] == "amount":
                details = timeperiod[Base_Constants.period["signal key value"]]
                if details is  None:
                    print("No valid time, please try again!")
                    return None

                while True:
                    # Wait for a connection
                    if(details == len(self.coalition_members)):
                        print("Wanted amount reached!")
                        return None
                    else:
                        print ("Wanted amount:", details, "amount:", len(self.coalition_members))
                    applicant = self.communication_protocoll.receive_information(self)
                    if(applicant.tag ==Base_Constants.possible_tags[5]): #requesting specs from coalition
                        coalition_info_message = Impl_Methods.Message(tag = Base_Constants.possible_tags[7], object=self.coalition)
                        self.communication_protocoll.send_information(self, coalition_info_message, communication_address=applicant.ip)
                    elif applicant.tag == Base_Constants.possible_tags[6]: #Tag needs to be apply
                        if(applicant.object in self.application_list):
                            pass
                        else:
                            return applicant, None
        elif timeperiod is not None and timeperiod[Base_Constants.period["phase"]] == "Always":
            pass


class applyForCoalition():
    def applyForCoalition(self):
        """
        time is the time the applicant waits for a response from the configurator or coordinator
        """
        application_object = Impl_Methods.CoalitionEntry(self.id, self.ip, self.agent)
        application_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[6], object=application_object)
        self.communication_protocoll.send_information(self, message=application_message, communication_address = self.server_address)
            

class selectAgent():
    def selectAgent(self):
        """
        Since every selector is also coordinator, he has to have an own selection of clients
        Selector activity
        we return the pairings where both, the trainer and updater have stated their readiness
        """
        selection_trainer = []
        selection_updater = []
        selected_pair = []

        for member in self.readyAgents:
            if "trainer" in member.agent.roles:
                selection_trainer.append(member)
            if "updater" in member.agent.roles:
                selection_updater.append(member)

        for pair in self.TraUpdPairing:
            for updater in selection_updater:
                if updater.id == pair[1].id:
                    for trainer in selection_trainer:
                        if trainer.id == pair[0].id:
                            selected_pair.append(pair)
                            break
                    break
        self.selected_Agents = selected_pair
        print("Selected Agents: "+str(len(self.selected_Agents)))



class announceAgentSelection():
    def announceAgentSelection(self):
        """
        role is either update or train
        agent_selection is created in selectAgent
        Selector activity
        """
        for pair in self.selected_Agents:
            selected_trainer_object = Impl_Methods.CoalitionEntryResponse(True, pair[0].id, pair[0].ip, "trainer") #trainer gets confirmation he has been chosen
            selected_updater_object = Impl_Methods.CoalitionEntryResponse(True, pair[0].id, pair[0].ip, "updater") #we send updater the agent he will have to update
            selected_trainer_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[9], object= selected_trainer_object)
            selected_updater_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[9], object= selected_updater_object)
            print(selected_updater_object.id, selected_updater_object.agent)
            if pair[0].id == self.id:
                #Could be a TraUpd is only Upd --> we note what action we are supposed to be
                self.actions[selected_trainer_object.agent] = True
            else:
                self.communication_protocoll.send_information(self, selected_trainer_message, communication_address = pair[0].ip)

            if pair[1].id == self.id:   
                if selected_trainer_object.id in self.agent_ready_to_update:
                    self.agent_ready_to_update.remove(selected_trainer_object.id)
                self.agent_ready_to_update.append(selected_trainer_object.id)
                #Could be a TraUpd is only Upd --> we note what action we are supposed to be
                self.actions[selected_updater_object.agent] = True
            else:
                self.communication_protocoll.send_information(self, selected_updater_message, communication_address = pair[1].ip)
            
        for agent in self.readyAgents:
            #After every chosen Agent has been notified they get a message to move to the next step
            if self.id != agent.id:
                done_message = Impl_Methods.Message(tag = Base_Constants.possible_tags[12], object=Base_Constants.possible_tags[12])
                self.communication_protocoll.send_information(self, done_message, communication_address = agent.ip)


class awaitSelectionSignal():
    def awaitSelectionSignal(self):
        """
        This method waits for the Selector to decide wether a client has been chosen or not for the next round of training.
        time is not the same as in await application, here a client just states how long he would wait in seconds
        """
        while True:
            selector_response = self.communication_protocoll.receive_information(self)
            if selector_response.tag == Base_Constants.possible_tags[9]:
                if selector_response.object.agent == "updater":

                    if selector_response.object.id in self.agent_ready_to_update:
                        self.agent_ready_to_update.remove(selector_response.object.id)

                    self.agent_ready_to_update.append(selector_response.object.id)

                #Could be a TraUpd is only Upd --> we note what action we are supposed to be
                self.actions[selector_response.object.agent] = True
            elif selector_response.tag == Base_Constants.possible_tags[12]:
                break
        


class trainMLModel():
    def trainMLModel(self, epochs=None):
        """
        In this method a Trainer Agent can train their local Ml model.
        """
        # Define loss and optimizer
        if epochs is None:
            epochs = 1
        print(f"Training {epochs} epoch(s) w/ {len(self.trainloader)} batches each")
        #print(self.MLTask)
        optimizer = self.model_optimizer
        self.ML_Model.train()
        # Train the network
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.ML_Model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                    self.training_loss.append(running_loss / 2000)
                    running_loss = 0.0
        self.interimResult = copy.deepcopy(self.ML_Model)

class awaitReadiness():
    """
        In this method a selector agents waits for the assigned agents to signal their readiness
        """
    def awaitReadiness(self, timer=None, amount=1):
        if timer is not None:
            start_time = time.time()
            while time.time() - start_time < timer and (amount==1 or amount != len(self.readyAgents)):
                message = self.communication_protocoll.receive_information(self)
                if(message.tag == Base_Constants.possible_tags[10]):
                    self.readyAgents.append(message.object)
        else:
            while amount != len(self.readyAgents):
                time.sleep(2)
                message = self.communication_protocoll.receive_information(self)
                if(message.tag == Base_Constants.possible_tags[10]):
                    self.readyAgents.append(message.object)

class signalReadiness():
    def signalReadiness(self):
        """
        In this method a Trainer Agent or an Updater Agent signal their readiness to the selector agent
        the message is a True with their id, ip and the agent they are
        """
        if self.selector.id == self.id:
            ready_object = Impl_Methods.CoalitionEntryResponse(True, self.id, self.ip, self.agent)
            self.readyAgents.append(ready_object)
        else:
            ready_object = Impl_Methods.CoalitionEntryResponse(True, self.id, self.ip, self.agent)
            ready_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[10], object=ready_object)
            self.communication_protocoll.send_information(self, message=ready_message, communication_address = self.selector.ip)

class transmitInterimResult():
    def transmitInterimResult(self):
        """
        In this method a Trainer Agent or an Updater Agent transmit the created interim results to a chosen updater.
        """
        if self.id == self.updater.id:
            pass
        else:
            interim_result_object = (self.interimResult, len(self.trainloader))
            interim_result_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[11], object=interim_result_object)
            self.communication_protocoll.send_information(self, message=interim_result_message, communication_address=self.updater.ip)

class updateMLModel():
    def updateMLModel(self):
        """
        In this method a updater agent updates the local model it has and then provides the update to the assigned agents if necessary.
        """

        global_model_ = copy.deepcopy(self.receivedInterimResults[0][0])

        total_size = sum(size for _, size in self.receivedInterimResults)
    
        # Initialize the global model with zeros (copy structure from the first model)
        global_model = {key: torch.zeros_like(value, dtype=torch.float32) 
                    for key, value in self.receivedInterimResults[0][0].state_dict().items()}
    
        # Weighted aggregation
        for model, size in self.receivedInterimResults:
            weight = size / total_size
            for key, value in model.state_dict().items():
                global_model[key] += weight * value.to(dtype=torch.float32)
        # Load averaged parameters back into the global model
        global_model_.load_state_dict({
        key: value.to(dtype=global_model_.state_dict()[key].dtype)
        for key, value in global_model.items()})
        
        for agent in self.agent_ready_to_update:
            if agent == self.id:
                self.ML_Model = global_model_
                self.ML_Model.to(self.device)
            else:
                for trainer in self.trainers:
                    if trainer.id == agent:
                        update_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[15], object=copy.deepcopy(global_model_))
                        self.communication_protocoll.send_information(self, message=update_message, communication_address=trainer.ip)
                        break
        self.receivedInterimResults = []

class informApplicant():
    def informApplicant(self, agent_selection, agent_not_selected):
        """
        In this method a the Configurator or Coordinator informs applicants if they have been chosen for training.
        """
        #Send a message to the adress of the applicants, that they were taken and what there role will be
        for applicant in agent_selection:
            print("Accepting Applicant!")
            if(applicant.ip != self.ip):
                application_object = Impl_Methods.CoalitionEntryResponse(True, self.id, self.ip, applicant.agent)
                communication_address= applicant.ip
                applicant_information_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[8], object=application_object)
                self.communication_protocoll.send_information(self, applicant_information_message, communication_address=communication_address)
        for applicant in agent_not_selected:
            print("Declining Applicant!")
            if(applicant.ip != self.ip):
                application_object = Impl_Methods.CoalitionEntryResponse(False, self.id, self.ip, applicant.agent)
                communication_address=applicant.ip
                applicant_information_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[8], object=application_object)
                self.communication_protocoll.send_information(self, applicant_information_message, communication_address = communication_address)

class provideMLTask():
    def provideMLTask(self, agent):
        """
        In this method a configurator or coordinator agent send the ML task to and applicant.
        """
        #Send a message to all agents, that take part, what is going to be trained
        print("Send ML Task!")
        if(agent.ip != self.ip):
            model = self.MLTask
            communication_address=agent.ip
            ML_Task_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[2], object=model)
            self.communication_protocoll.send_information(self, ML_Task_message, communication_address=communication_address)
    
class registerCoalition():
    def registerCoalition(self, purposeOfSystem=None, agentRequirements=None):
        """
        Defines and returns the following three parameters
        purposeOfSystem means the general prediction problem that should be solved
        agentRequirements means, things as to what the agents need to be, what devices they have, how much and how high quality data they should have etc.
        the agent Requirements are suppossed to be in form of Impl_Methods.AgentRequirements
        repository means the CDML system specifications
        """
        if purposeOfSystem is None:
            purposeOfSystem = Impl_Methods.PurposeOfSystem()
        if agentRequirements is None:
            agentRequirements = Impl_Methods.AgentRequirements()
        if self.repository is None:
            #If there isn't a specific purpose and agent requirements don't exist we return the basic objects which state that it's not defined. The repository however is mandatory!
            print("You shouldn't registere a coalition without a repository!")
            return None
        
        coalition = Impl_Methods.Coalition(purposeOfSystem, agentRequirements, self.repository)
        self.coalition = coalition

class awaitUpdate():
    """
        This method is new to the original concept and lets the trainer await an update by the updater.
    """
    def awaitUpdate(self):
        if self.id != self.updater.id:
            while True:
                update = self.communication_protocoll.receive_information(self)
                if update.tag == Base_Constants.possible_tags[15]:
                    self.ML_Model = update.object
                    self.ML_Model.to(self.device)
                    break

class decideOnApplication():
    def decideOnApplication(self, application):
        """
        The base method accepts every applicant for the agents of trainer. Every other role, will be only once given away, so if  
        """
        base_amount_of_roles = Base_Constants.base_amount_of_roles
        current_num_of_roles = Base_Constants.initialization_amount_of_roles #TODO num of open roles should be stored at the Configurator
        agent_selection = []
        agent_not_selected = []

        allowed = True
        for role in application.agent.roles:
            if int(base_amount_of_roles[role]) <= int(current_num_of_roles[role]):
                allowed = False
                agent_not_selected.append(application)
                break
        if allowed:
            agent_selection.append(application)
            for role in application.agent.roles:
                current_num_of_roles[role] = str(int(current_num_of_roles[role]) + 1)
        return agent_selection, agent_not_selected

class defineInterimResults():
    def defineInterimResults(self, interimResults = None):
        """
        Interim results are updates which are computed by agents based on local training data.
        """
        if(interimResults == None):
            print("Interim results need to be defined, there is no standard interim result in this framework!"+
                  "\n"+
                  "Choose the returned dict for structure!")
            return Impl_Methods.InterimResult 
            #here we return the class and not an object of the class. 
            # Agents should then create objects of the class itself to create interim results
        else:
            self.interimResults = interimResults

class defineInitialMLModel():
    def defineInitialMLModel(self, model=None):
        """
        Is set by the configurator
        Can be information about the (first) layers of neural networks, a (sub-) set of parameters of linear regression, activation functions, and the ML model architecture
        """
        if model == None:
            print("Model need to be defined, there is no standard model in this framework!"+
                  "\n"+
                  "The returned model can be used as an example!")
            return Base_Model.resnet18()
        else:
            self.ML_Model = model


class testMLModel():
    def testMLModel(self):
        """
        A basic test method allowing the test of the trained model.
        """
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        criterion = nn.CrossEntropyLoss()
        testset = torchvision.datasets.CIFAR100(root='./data_cifar100', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        model = self.ML_Model
        model.eval
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        print(f'Test Loss: {test_loss/len(testloader)}')
        print(f'Accuracy: {100 * correct / total}%')
        return f'Test Loss: {test_loss/len(testloader)}', f'Accuracy: {100 * correct / total}%'


class defineRepository():
    def defineRepository(self, repository=None):
        """
        Allows the configurator the set up a repository which applicants can fetch.
        """
        if repository == None:
            self.repository = self.interimResultsDefinition
        else:
            self.repository = repository

class assignInterimResultRecipient():
    def assignInterimResultRecipient(self, trainer):
        """
        Here we exploit the fact, that every selector is also a coordinator. With this fact we can say that each selector only deals with agents and updaters that he himself grouped
        together. If a selector would just have to get updaters and trainers assigned we couldnt be sure that the updaters and trainers match and we cant ensure that the right updater
        and trainers are ready. But here we assume that coordinators that assign trainers to updaters that they stay the selector for besad pairings.
        """
        updater = self.updater_agents[random.randint(0,len(self.updater_agents)-1)] #choose a random updater and assign it to the trainer and the trainer to the updater
        self.TraUpdPairing.append((trainer,updater))
        trainer_comm_addr = trainer.ip
        updater_comm_addr = updater.ip
        if updater.id == self.id and trainer.id == self.id: #The CooUpdTra combination is the updater -> we don't have to send data we just store it
            time.sleep(1)
            self.trainers.append(trainer)
            self.updater = updater
        elif updater.id == self.id: #The CooUpd combination is the updater -> we don't have to send data we just store it
            time.sleep(1)
            self.trainers.append(trainer)
            interim_result_recipient_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[4], object=updater)
            self.communication_protocoll.send_information(self, interim_result_recipient_message, communication_address=trainer_comm_addr)
        elif trainer.id == self.id: #The CooTra combination is trainer -> we don't have to send data we just store it 
            time.sleep(1)
            self.updater = updater
            interim_result_recipient_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[3], object=trainer)
            self.communication_protocoll.send_information(self, interim_result_recipient_message, communication_address=updater_comm_addr)
        else: #We do not have to ensure CooTraUpd works, since this Combination would already be checked in config since with CooTraUpd the first member would already be Tra and Upd
            time.sleep(1)
            interim_result_recipient_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[4], object=updater)
            self.communication_protocoll.send_information(self, interim_result_recipient_message, communication_address=trainer_comm_addr)
            time.sleep(1)
            interim_result_recipient_message = Impl_Methods.Message(tag=Base_Constants.possible_tags[3], object=trainer)
            self.communication_protocoll.send_information(self, interim_result_recipient_message, communication_address=updater_comm_addr)

class assignSelectorAgent():
    def assignSelectorAgent(self):
            """
        In this method a selector agent is assigned to trainer and updater.
        """
            for member in self.coalition_members:
                if "selector" in member.agent.roles:
                    selector = member
                    break

            for member in self.coalition_members:
                if member.id == self.id:
                    self.selector = selector
                elif member.id == selector.id:
                    message = Impl_Methods.Message(tag = Base_Constants.possible_tags[14], object = self.TraUpdPairing)
                    self.communication_protocoll.send_information(self, message=message, communication_address=member.ip)
                else:
                    message = Impl_Methods.Message(tag = Base_Constants.possible_tags[13], object = selector)
                    self.communication_protocoll.send_information(self, message=message, communication_address=member.ip)


class awaitInterimResults():
    """
        In this method a updater agents wait until they receive interim results.
        """
    def awaitInterimResults(self):
        self.receivedInterimResults = []
        if "trainer" in self.agent.roles and self.updater.id == self.id:
            self.receivedInterimResults.append((self.interimResult, len(self.trainloader)))
        while len(self.receivedInterimResults)!=len(self.agent_ready_to_update):
            message = self.communication_protocoll.receive_information(self)
            if(message.tag == Base_Constants.possible_tags[11]):
                self.receivedInterimResults.append(message.object)