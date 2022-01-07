from GameAdapter import GameAdapter, Player
from LCS_Actor import LCSActor
from LCS_Rules import LCSRules


class LCSPlayer(Player):

    def __init__(self, name, rules: LCSRules):
        super().__init__(name)
        self.rules = rules
        self.actor = None

    def setup(self):
        self.actor = LCSActor(self.io)

    def make_action(self, state):
        act_string = self.rules.act(state)
        self.actor.act(act_string)




