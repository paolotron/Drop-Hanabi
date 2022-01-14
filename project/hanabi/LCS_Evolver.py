from typing import List
from threading import Thread
from multiprocessing import Process
from LCS_Rules import LCSRules, RuleSet
from LCS_Actor import LCSActor
import server_custom as server
from LCS_Rules import RuleSet
from player_LCS import LCSPlayer
import LCS_Sensor as sens
from constants import *


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
        self.serv = Process(target=server.start_server, args=[self.n_players, PORT])
        self.serv.start()
        self.players = [LCSPlayer(name=f'LCS{i}') for i in range(self.n_players)]

    @staticmethod
    def action_len():
        return LCSActor.get_action_length()

    def sensor_len(self):
        return sens.get_sensor_len(self.n_players)

    def get_fitness(self, rule_list: List[LCSRules], sensors=None):
        if sensors is None:
            sensors = sens.package_sensors(self.n_players)
        lcs_rule_list = [LCSRules(sensors, LCSActor.get_action_length(), rule) for rule in rule_list]
        threads = [Thread(target=player.start, args=[lcs_rule_list[i]]) for i, player in enumerate(self.players)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        results = [player.end_game_data() for player in self.players]
        data = [player.rules.end_game_data() for player in self.players]
        res = {
            "n_turns": max(map(lambda x: x["n_turns"], results)),
            "points": max(map(lambda x: x["points"], results)),
            "loss": max(map(lambda x: x["loss"], results)),
            "rule_match": [d.rule_match for d in data],
            "critical_rules": [d.critical_rules for d in data],
            "rule_use": [d.rule_usage for d in data]
        }

        return res


def dummy_play(n_players):
    man = GameManager(n_players)
    for i in range(100):
        r1 = RuleSet.random_rule_set(man.sensor_len(), man.action_len(), 100)
        r2 = RuleSet.random_rule_set(man.sensor_len(), man.action_len(), 100)
        j = man.get_fitness([r1, r2])
        if i % 10 == 0:
            print(i)


if __name__ == '__main__':
    n_players = 5
    g = GameManager(n_players)
    r1 = RuleSet.random_rule_set(g.sensor_len(), g.action_len(), 100)
    r2 = RuleSet.random_rule_set(g.sensor_len(), g.action_len(), 20)
    r3 = RuleSet.random_rule_set(g.sensor_len(), g.action_len(), 1)
    r4 = RuleSet.random_rule_set(g.sensor_len(), g.action_len(), 50)
    r5 = RuleSet.random_rule_set(g.sensor_len(), g.action_len(), 40)
    res = g.get_fitness([r1, r2, r3, r4, r5])
    print(res)