import torch
from ...base import Base_Roles
from . import AdaptedMethods
from ...base import Impl_Methods

class configurator(Base_Roles.basic_configurator):
    pass

class coordinator(Base_Roles.basic_coordinator):
    pass

class selector(Base_Roles.basic_selector):
    pass


class trainer(Base_Roles.basic_trainer, AdaptedMethods.SplitLearningVanilla):
    """
    Is adapted in trainMLModel, awaitUpdate and testMLModel to accomodate Vanilla Split Learning.
    """
    def trainMLModel(self, epochs=None):
        AdaptedMethods.SplitLearningVanilla.trainMLModel(self,epochs)
    def awaitUpdate(self):
        AdaptedMethods.SplitLearningVanilla.awaitUpdate(self)
    def testMLModel(self):
        AdaptedMethods.SplitLearningVanilla.testMLModel(self)


class updater(Base_Roles.basic_updater):
    """
    Is adapted in init and updateMLModel to accomodate Vanilla Split Learning.
    """
    def __init__(self, id=-1, communication_protocoll=None,ip=None, agent = "Upd", server_address=None,ML_Model_update=None,device=None,update_optimizer=None, **kwargs):
        self.chosen = False
        self.selector = None
        self.trainers = []
        self.actions = {"trainer":False, "updater": False}
        self.agent_ready_to_update = []
        self.receivedInterimResults = []
        self.ML_Model_update = ML_Model_update
        if update_optimizer is not None:
            self.update_optimizer = update_optimizer(self.ML_Model_update.parameters(), lr=0.1, momentum=0.9)
        else:
            self.update_optimizer= torch.optim.SGD(self.ML_Model_update.parameters(), lr=0.1, momentum=0.9)

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

    def updateMLModel(self):
        AdaptedMethods.SplitLearningVanilla.updateMLModelSplitLearning(self)

