from typing import Dict, List

import numpy as np
from game import Card
from numpy.typing import ArrayLike
from knowledge import KnowledgeMap, Color


def __evaluate_card(probability: ArrayLike, table_cards: Dict[str, List[Card]]) -> int:
    """
    return -1 if a card is discardable, 1 if it is playable and 0 if not have enough information
    @param probability: matrix 5x5
    @return: integer
    """
    p = []
    for color in table_cards.keys():
        index = Color.fromstr(color)
        if len(table_cards[color]) < 5:
            p.append(probability[index][table_cards[color][-1].value])
        for card in table_cards[color]:
            if probability[index][card.value] == 1:
                # discard
                return -1
    if any([x >= 1 for x in p]):
        # playable
        return 1
    # not enough information on the card
    return 0


def discard_known(my_hand: list[ArrayLike], table_cards: Dict[str, list[Card]]) -> list[bool]:
    """
    return the bitstring that represent which card of the hand are sure discard
    @param my_hand: list of matrix 5x5
    @param table_cards: card on the table
    @return list of booleans
    """
    ret = []
    for i, my_knowledge_matrix in enumerate(my_hand):
        if __evaluate_card(my_knowledge_matrix, table_cards) == -1:
            ret.append(True)
        else:
            ret.append(False)
    return ret


def discard_unknown(my_hand: list[ArrayLike]) -> list[bool]:
    """
    return the bitstring that represent which card of the hand are possible discard
    @param my_hand: list of matrix 5x5
    @return list of booleans
    """
    ret = []
    for i, my_knowledge_matrix in enumerate(my_hand):
        if np.any(np.sum(my_knowledge_matrix, axis=0)[:] >= 1) or np.any(np.sum(my_knowledge_matrix, axis=1)[:] >= 1):
            ret.append(False)
        else:
            ret.append(True)
    return ret


def play_known(my_hand: list[ArrayLike], table_cards: Dict[str, list[Card]]) -> list[bool]:
    """
    return the bitstring that represent which card of the hand are sure play
    @param my_hand: list of matrix 5x5
    @param table_cards: cards on the table
    @return list of booleans
    """
    ret = []
    for i, my_knowledge_matrix in enumerate(my_hand):
        if __evaluate_card(my_knowledge_matrix, table_cards) == 1:
            ret.append(True)
        else:
            ret.append(False)
    return ret


def play_unknown(my_hand: list[ArrayLike]):
    """
    return the bitstring that represent which card of the hand are possible play
    @param my_hand: list of matrix 5x5
    @return list of booleans
    """
    ret = []
    for i, my_knowledge_matrix in enumerate(my_hand):
        if np.any(np.sum(my_knowledge_matrix, axis=1)[i] >= 1):
            ret.append(True)
        else:
            ret.append(False)
    return ret


def __can_be_played(card: Card, table_cards: Dict[str, list[Card]]) -> bool:
    """
    return if a card can be played or not
    @param card: Card
    @param table_cards: cards on the table
    @return True if is playable or False on the contrary
    """
    last_card = table_cards[card.color]
    if not last_card:
        return card.value == 1
    return last_card[-1].value + 1 == card.value


def __hint_type(knowledge: ArrayLike, card: Card) -> int:
    """
    return information about card on the end of a player
    @param knowledge: matrix 5x5
    @param card: card
    @return -1 if player doesn't know anything about the card
    @return 0 if player already knows everything
    @return 1 if player knows just the number
    @return 2 if player knows just the color
    """
    index = Color.fromstr(card.color)
    res1 = False
    res2 = False
    # player already knows the number
    if np.sum(knowledge, axis=0)[card.value - 1] >= 1:
        res1 = True
    # player already knows the color
    if np.sum(knowledge, axis=1)[index] >= 1:
        res2 = True

    if res1 and res2:
        return 0
    if res1:
        return 1
    if res2:
        return 2
    return -1


def hint_number(knowledge_map: KnowledgeMap) -> list[bool]:
    """
    return the bitstring that represent which number are hintable
    @param knowledge_map: KnowledgeMap
    @return list of booleans
    """
    def check_number(n: int, hand: list[Card], table_cards: Dict[str, list[Card]]):
        ret_val = False
        for card in hand:
            if card.value == n and __can_be_played(card, table_cards):
                ret_val = True
            if card.value == n and not __can_be_played(card, table_cards):
                return False
        return ret_val

    ret = []
    for player in knowledge_map.getPlayerList():
        for i in range(1, 6):
            val = check_number(i, player.hand, knowledge_map.getTableCards())
            ret.append(val)

    return ret


def hint_color(knowledge_map: KnowledgeMap):
    """
    return the bitstring that represent which color are hintable
    @param knowledge_map: KnowledgeMap
    @return list of booleans
    """
    def check_color(col: Color, hand: list[Card], table_cards: Dict[str, list[Card]]):
        ret_val = False
        for card in hand:
            if card.color == col and __can_be_played(card, table_cards):
                ret_val = True
            if card.value == col and not __can_be_played(card, table_cards):
                return False
        return ret_val

    ret = []
    for player in knowledge_map.getPlayerList():
        for color in Color:
            if color == Color.UNKNOWN:
                continue
            val = check_color(color, player.hand, knowledge_map.getTableCards())
            ret.append(val)

    return ret


def hint_discard(knowledge_map: KnowledgeMap):
    """
    return the bitstring that represent which card are hintable to discard
    @param knowledge_map: KnowledgeMap
    @return list of booleans
    """
    def number_can_be_discarded(value: int, table_cards: Dict[str, list[Card]]) -> bool:
        for cards_in_table in table_cards.values():
            if value > len(cards_in_table):
                return False
        return True

    def color_can_be_discarded(c: str, table_cards: Dict[str, list[Card]]) -> bool:
        return len(table_cards[c]) == 5

    ret = [False] * 10 * len(knowledge_map.getPlayerList())
    j = 0
    for player in knowledge_map.getPlayerList():
        for card in player.hand:
            ret[j + card.value - 1] = number_can_be_discarded(card.value, knowledge_map.getTableCards())
            ret[j+5 + Color.fromstr(card.color)] = color_can_be_discarded(card.color, knowledge_map.getTableCards())
        j += 10

    return ret
