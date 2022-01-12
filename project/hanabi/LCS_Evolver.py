from typing import List, Union
import os
import time
from threading import Thread, Semaphore
import numpy as np

from LCS_Rules import LCSRules, RuleSet
from LCS_Actor import LCSActor
import server_custom as server
from LCS_Rules import RuleSet
from player_LCS import LCSPlayer
import LCS_Sensor as sens


class Evolver:

    def __init__(self, lcs_rules_list: List[RuleSet]):
        self.LCS_rules_list = lcs_rules_list

    def evolve(self, activations, fitness):
        # Activations arriva dal metodo EndGameData
        # sto metodo mi ritorna altri ruleSet
        pass

class GameManager:

    def __init__(self, n_players):
        self.n_players = n_players
        self.serv = Thread(target=server.start_server, args=[self.n_players])
        self.serv.start()
        self.players = [LCSPlayer(name=f'LCS{i}') for i in range(self.n_players)]

    def get_fitness(self, rule_list: List[LCSRules]):
        threads = [Thread(target=player.start, args=[rule_list[i]]) for i, player in enumerate(self.players)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        results = [player.io.end_game_data() for player in self.players]
        return results

    def close(self):
        self.serv.join()


def dummy_play(n_players):
    man = GameManager(n_players)
    for i in range(100):
        rules = [LCSRules(sens.package_sensors(n_players), LCSActor.get_action_length()) for _ in range(n_players)]
        j = man.get_fitness(rules)
        res = [rule.end_game_data() for rule in rules]
    return

if __name__ == '__main__':
    npl = 3
    dummy_play(npl)
