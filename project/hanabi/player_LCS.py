from GameAdapter import Player
from LCS_Actor import LCSActor
from LCS_Rules import LCSRules


class LCSPlayer(Player):

    def __init__(self, name):
        super().__init__(name)
        self.rules = None
        self.actor = None

    def setup(self, rules: LCSRules):
        self.rules = rules
        self.actor = LCSActor(self.io)

    def make_action(self, state):
        act_string = self.rules.act(state)
        self.actor.act(act_string)




