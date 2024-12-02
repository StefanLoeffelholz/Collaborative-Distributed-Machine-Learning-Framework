from ...base import Base_Roles
from . import AdaptedMethods
from ...base import Base_Methods
from ...base import Base_Constants

class configurator(Base_Roles.basic_configurator):
    pass

class coordinator(Base_Roles.basic_coordinator):
    pass

class selector(Base_Roles.basic_selector):
    pass


class trainer(Base_Roles.basic_trainer, AdaptedMethods.GradientAssistedLearning):
    """
    Is specified in trainMLModel, awaitUpdate and operation phase to accomodate GAL.
    """
    def trainMLModel(self, epochs=None):
        AdaptedMethods.GradientAssistedLearning.trainMLModelAssisted(self,epochs)
    def awaitUpdate(self):
        AdaptedMethods.GradientAssistedLearning.awaitUpdateAssisted(self)
    def operation_phase(self, rounds=1):
        for i in range(0,rounds):
            if self.actions["trainer"]==False:
                self.signalReadiness()
                self.awaitSelectionSignal()
            self.awaitUpdate()
            self.trainMLModel(epochs=2)
            self.transmitInterimResult()
            self.testMLModel()
            while True:
                message = self.communication_protocoll.receive_information(self)
                if message.tag == Base_Constants.possible_tags[12]:
                    break
            self.actions["trainer"]=False


class updater(Base_Roles.basic_updater, Base_Methods.testMLModel):
    """
    Is specified in updateMLModeland operation phase to accomodate GAL.
    """

    def updateMLModel(self):
        AdaptedMethods.GradientAssistedLearning.updateMLModelAssistedLearning(self, epochs = 2)
    def operation_phase(self, rounds=1):
        for i in range(0,rounds):
            if self.actions["updater"]==False:
                self.signalReadiness()
                self.awaitSelectionSignal()
            self.updateMLModel()
            Base_Methods.testMLModel.testMLModel(self)
            self.awaitInterimResults()
            self.actions["updater"]=False
            self.agent_ready_to_update = []

