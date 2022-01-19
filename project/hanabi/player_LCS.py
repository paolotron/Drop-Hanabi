from typing import Union

from GameAdapter import Player
import LCS_Actor as Act
import LCS_Rules as Rul
import LCS_Sensor as Sen
from sys import argv
import numpy as np


class LCSPlayer(Player):

    def __init__(self, name):
        super().__init__(name)
        self.rules: Union[Rul.LCSRules, None] = None
        self.actor = None

    def setup(self, rules: Rul.LCSRules):
        self.rules: Rul.LCSRules = rules
        self.actor = Act.LCSActor(self.io)

    def make_action(self, state):
        while True:
            act_string = self.rules.act(state)
            res = self.actor.act(act_string)
            if not res:
                self.rules.signal_critical_failure()
            else:
                break


if __name__ == '__main__':
    if len(argv) != 3:
        raise TypeError("Wrong number of arguments, must call 'python player_LCS.py num_players name'")
    num_players = int(argv[2])
    name = argv[1]
    rule_matr = np.load(f"./models/ruleset_{num_players}.npy")
    rule = Rul.LCSRules(Sen.package_sensors(num_players),
                        Act.LCSActor.get_action_length(num_players),
                        Rul.RuleSet.unpack_rules(rule_matr, Sen.get_sensor_len(num_players)))
    player = LCSPlayer(name)
    player.start(rule)
    result = player.io.end_game_data()
    print(f"Ended with {result['points']} points")
