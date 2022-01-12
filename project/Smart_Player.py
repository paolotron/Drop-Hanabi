from collections import defaultdict
from itertools import count

import hanabi.game as game
from hanabi.GameAdapter import GameAdapter
from hanabi.GameAdapter import Color
from hanabi.constants import *
from hanabi.GameAdapter import HintType
from random import choice
import sys


def tuple_value(tup):
    if (Color.UNKNOWN, 0) == tup:
        return 0
    if (tup[0] == Color.UNKNOWN and tup[1] != 0) or (tup[0] != Color.UNKNOWN and tup[1] == 0):
        return 1
    return 2


def useful_unknwown_cards(man: GameAdapter):

    def can_be_played(card: game.Card) -> bool:
        last_card = man.board_state.tableCards[card.color]
        if not last_card:
            return card.value == 1
        return last_card.value[-1] + 1 == card.value

    options = []
    for player in man.board_state.players:
        if player.name == man.name:
            continue
        useful_card = [(i, card) for i, card in enumerate(player.hand) if can_be_played(card)]
        value_of_useful_card = [(player.name, i, card, tuple_value(man.knowledge_state[player.name][i])) for i, card in useful_card]
        options += value_of_useful_card

    if not options:
        any_player = [pl for pl in man.board_state.players if pl.name != man.name][0]
        return any_player.hand[0], any_player.name
    card = min(options, key=lambda x: x[3])

    return card[2], card[0]


def main(name='Paolo'):
    if len(sys.argv) > 1:
        name = sys.argv[1]
    start_dict = {
        'name': name,
        'ip': HOST,
        'port': PORT,
        'datasize': DATASIZE,
        'nplayers': NPLAYERS
    }
    manager = GameAdapter(**start_dict)
    players = manager.get_other_players()
    for state, move_history in manager:
        valid = [ix for ix, know in enumerate(manager.knowledge_state[name]) if know != (Color.UNKNOWN, 0)]

        if valid:
            manager.send_play_card(valid[0])
            continue

        if manager.board_state.usedNoteTokens == 8:
            manager.send_discard_card(0)
            continue

        card, name = useful_unknwown_cards(manager)
        manager.send_hint(name, HintType.NUMBER, card.value)


if __name__ == '__main__':
    main()
