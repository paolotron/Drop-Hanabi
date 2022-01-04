from hanabi.GameAdapter import GameAdapter
from hanabi.LCS.LCS_Actor import LCSRules
from hanabi.LCS.LCS_Rules import LCSRules
from hanabi.constants import *


def main(rule_system: LCSRules, name='LCS'):
    start_dict = {
        'name': name,
        'ip': HOST,
        'port': PORT,
        'datasize': DATASIZE,
        'nplayers': NPLAYERS
    }
    adapter = GameAdapter(**start_dict)
    actor = LCSRules(adapter)
    for knowledge_map in adapter:
        act_string = rule_system.act(knowledge_map)
        actor.act(act_string)



