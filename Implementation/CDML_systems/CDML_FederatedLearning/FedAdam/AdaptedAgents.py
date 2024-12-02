from ...base import Base_Agents
from ...base import Base_Roles as role
from . import AdaptedRoles

class Tra(Base_Agents.basic_Tra):
    def operation_phase(self, ready_agents=3, rounds=1):
        AdaptedRoles.trainer.operation_phase(self, rounds=1)
    
class CooSel(Base_Agents.basic_CooSel):
    pass

class TraUpd(Base_Agents.basic_TraUpd):
    pass

class ConTraUpd(Base_Agents.basic_ConTraUpd):
    pass

class ConCooSelUpd(Base_Agents.basic_ConCooSelUpd):
    pass

class CooSelTraUpd(Base_Agents.basic_CooSelTraUpd):
    pass

class ConCooSelTraUpd(Base_Agents.basic_ConCooSelTraUpd):
    def operation_phase(self, ready_agents=3, rounds=1):
        for i in range(0,rounds):
            print("Round: " + str(i))
            role.basic_selector.operation_phase(self, ready_agents=ready_agents, rounds=1)
            AdaptedRoles.trainer.operation_phase(self, rounds=1)
            role.basic_updater.operation_phase(self, rounds=1)





