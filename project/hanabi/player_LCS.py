from typing import Union
from GameAdapter import Player
import LCS_Actor as Act
import LCS_Rules as Rul
import LCS_Sensor as Sen
from sys import argv
import numpy as np


class LCSPlayer(Player):

    def __init__(self, name):
        """
        Initialize the player
        @param name:
        """
        super().__init__(name)
        self.rules: Union[Rul.LCSRules, None] = None
        self.actor = None

    def setup(self, rules: Rul.LCSRules):
        """
        Setup before starting a match
        @param rules: RuleSet to be used during play
        @return: None
        """
        self.rules: Rul.LCSRules = rules
        self.actor = Act.LCSActor(self.io)

    def make_action(self, state):
        """
        1. Get Environment
        2. Encode Environment into bitstring with Sensors
        3. Match Sensor output with RuleSet
        4. If no match found cover and goto 3.
        5. Choose rule to activate among those matched
        6. Get action string from matched rule
        7. Do action
        8. If action is invalid mark rule as invalid and goto 3.
        9. End turn

        An action is considered invalid if the server refuses it,
        cases could be invalid moves such as giving hints when no
        hint tokens are left
        @param state: KnowledgeMap
        @return: None
        """
        while True:
            act_string = self.rules.act(state)
            res = self.actor.act(act_string)
            if not res:
                self.rules.signal_critical_failure()
            else:
                break


def main(name, num_players):
    rule_matr = np.load(f"./models/ruleset_{num_players}.npy")
    rule = Rul.LCSRules(Sen.package_sensors(num_players),
                        Act.LCSActor.get_action_length(),
                        Rul.RuleSet.unpack_rules(rule_matr, Sen.get_sensor_len(num_players)))
    player = LCSPlayer(name)
    player.start(rule)
    result = player.io.end_game_data()
    print(f"Ended with {result['points']} points")


if __name__ == '__main__':
    # See README for usage info
    if len(argv) != 3:
        raise TypeError("Wrong number of arguments, must call 'python player_LCS.py num_players name'")

    main(argv[1], int(argv[2]))
