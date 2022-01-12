from typing import List
import os
import time
from threading import Thread, Semaphore

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


def get_fitness(rule_list: List[LCSRules]):
    threads = [Thread(target=player.start, args=[rule_list[i]]) for i, player in enumerate(players)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    results = [player.io.end_game_data() for player in players]
    return results


def dummy_play():
    for i in range(10):
        j = get_fitness([LCSRules(sensor_list=sens.package_sensors(n_players),
                                  action_length=LCSActor.get_action_length(2))
                         for _ in range(n_players)])
        pass


if __name__ == '__main__':
    n_players = 2
    server = Thread(target=server.start_server, args=[n_players])
    server.start()
    players = [LCSPlayer(name=f'LCS{i}') for i in range(n_players)]
    dummy_play()


