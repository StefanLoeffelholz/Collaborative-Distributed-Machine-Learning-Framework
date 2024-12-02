import copy
import torch
from ...base import Base_Methods
from ...base import Base_Constants

class awaitApplication(Base_Methods.awaitApplication):
    def awaitApplication(self):
        pass

class applyForCoalition(Base_Methods.applyForCoalition):
    def applyForCoalition(self):
        pass

class selectAgent(Base_Methods.selectAgent):
    def selectAgent(self):
        pass

class announceAgentSelection(Base_Methods.announceAgentSelection):
    def announceAgentSelection(self):
        pass

class awaitSelectionSignal(Base_Methods.awaitSelectionSignal):
    def awaitSelectionSignal(self):
        pass

class trainMLModel(Base_Methods.trainMLModel):
    def trainMLModel(self, epochs = 1, mu = 0.1):
        """Train the network."""
        # Define loss and optimizer
        mu = 0.01
        if epochs is None:
            epochs = 1
        print(f"Training {epochs} epoch(s) w/ {len(self.trainloader)} batches each")
        #print(self.MLTask)
        optimizer = self.model_optimizer
        self.ML_Model.train()
        global_weights = self.ML_Model.state_dict()
        global_weights_tensor = {name: torch.clone(param).detach().to(self.device) 
                                    for name, param in global_weights.items()}
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
                proximal_term = 0.0
                for name, param in self.ML_Model.named_parameters():
                    proximal_term += ((param - global_weights_tensor[name]) ** 2).sum()
                
                loss += (mu / 2) * proximal_term
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                    self.training_loss.append(running_loss / 2000)
                    running_loss = 0.0
        self.interimResult = copy.deepcopy(self.ML_Model)

class signalReadiness(Base_Methods.signalReadiness):
    def signalReadiness(self):
        pass

class transmitInterimResult(Base_Methods.transmitInterimResult):
    def transmitInterimResult(self):
        pass

class updateMLModel(Base_Methods.updateMLModel):
    def updateMLModel(self):
        pass

class informApplicant(Base_Methods.informApplicant):
    def informApplicant(self):
        pass

class provideMLTask(Base_Methods.provideMLTask):
    def provideMLTask(self):
        pass

class registerCoalition(Base_Methods.registerCoalition):
    def registerCoalition(self):
        pass

class decideOnApplication(Base_Methods.decideOnApplication):
    def decideOnApplication(self):
        pass

class defineInterimResults(Base_Methods.defineInterimResults):
    def defineInterimResults(self):
        """
        Here we create an interim result which can be given in the repository to people, in Federated Learning we always share the entire model so the phrase "model" is the result
        """
        result = Base_Constants.possible_results["Federated"]
        details = None
        self.interim_result = (result, details)

class defineInitialMLModel(Base_Methods.defineInitialMLModel):
    def defineInitialMLModel(self, model=None):
        if model is not None:
            self.model = model

class assignInterimResultRecipient(Base_Methods.assignInterimResultRecipient):
    def assignInterimResultRecipient(self):
        pass

class awaitInterimResults(Base_Methods.awaitInterimResults):
    def awaitInterimResults(self):
        pass




