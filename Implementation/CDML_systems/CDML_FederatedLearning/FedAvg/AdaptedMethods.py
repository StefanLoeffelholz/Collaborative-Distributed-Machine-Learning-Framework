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
    def trainMLModel(self):
        pass

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




