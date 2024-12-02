from . import AdaptedMethods
from ...base import Base_Agents


class Tra(Base_Agents.basic_Tra):
    def trainMLModel(self, epochs = 2, mu = 0.01):
        return AdaptedMethods.trainMLModel.trainMLModel(self, epochs = epochs, mu=mu)
    
class CooSel(Base_Agents.basic_CooSel):
    pass

class TraUpd(Base_Agents.basic_TraUpd):
    def trainMLModel(self, epochs = 2, mu = 0.01):
        return AdaptedMethods.trainMLModel.trainMLModel(self, epochs = epochs, mu=mu)

class ConTraUpd(Base_Agents.basic_ConTraUpd):
    def trainMLModel(self, epochs = 2, mu = 0.01):
        return AdaptedMethods.trainMLModel.trainMLModel(self, epochs = epochs, mu=mu)

class ConCooSelUpd(Base_Agents.basic_ConCooSelUpd):
    pass

class CooSelTraUpd(Base_Agents.basic_CooSelTraUpd):
    def trainMLModel(self, epochs = 2, mu = 0.01):
        return AdaptedMethods.trainMLModel.trainMLModel(self, epochs = epochs, mu=mu)

class ConCooSelTraUpd(Base_Agents.basic_ConCooSelTraUpd):
    def trainMLModel(self, epochs = 2, mu = 0.01):
        return AdaptedMethods.trainMLModel.trainMLModel(self, epochs = epochs, mu=mu)
    





