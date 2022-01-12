from typing import List

from LCS.LCS_Rules import RuleSet

class Evolver:

    def __init__(self, LCS_rules_list: List[RuleSet]):
        self.LCS_rules_list = LCS_rules_list


    def evolve(self, activations, fitness): # Activations arriva dal metodo EndGameData
        # sto metodo mi ritorna altri ruleSet
        pass