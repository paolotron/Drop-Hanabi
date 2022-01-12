from typing import Union

from GameAdapter import Player
from LCS_Actor import LCSActor
from LCS_Rules import LCSRules


class LCSPlayer(Player):

    def __init__(self, name):
        super().__init__(name)
        self.rules: Union[LCSRules, None] = None
        self.actor = None

    def setup(self, rules: LCSRules):
        self.rules: LCSRules = rules
        self.actor = LCSActor(self.io)

    def make_action(self, state):
        while True:
            act_string = self.rules.act(state)
            res = self.actor.act(act_string)
            if not res:
                self.rules.signal_critical_failure()
            else:
                break





