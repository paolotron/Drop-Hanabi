from abc import ABC, abstractmethod
from numpy.typing import NDArray
from numpy import bool_
import numpy as np
from typing import Dict, List
from numpy.typing import ArrayLike
from knowledge import KnowledgeMap, Color


class GenericSensor(ABC):
    """
    Abstract Class that must be inherited for creating a new sensor
    """

    def __init__(self, out_size: int):
        """
        Constructor for Generic Sensor
        @param out_size: length of the outputted bit_string
        """
        self.out_size = out_size

    def get_out_size(self):
        return self.out_size

    @abstractmethod
    def activate(self, knowledge_map: KnowledgeMap) -> NDArray[bool_]:
        """
        Activate method that must be overloaded
        @param knowledge_map: Environment descriptor
        @return: bool array of 'self.out_size' length
        """
        pass

    def get_activate(self, knowledge_map: KnowledgeMap):
        res = self.activate(knowledge_map)
        assert res.size == self.out_size
        return res


class DiscardKnowSensor(GenericSensor):
    def __init__(self, n_player: int):
        if n_player == 2 or n_player == 3:
            super().__init__(5)
        else:
            super().__init__(4)

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(discard_known(knowledge_map.getProbabilityMatrix(knowledge_map.getPlayerName()),
                                      knowledge_map.getTableCards()))


class DiscardUnknownSensor(GenericSensor):
    def __init__(self, n_player: int):
        if n_player == 2 or n_player == 3:
            super().__init__(5)
        else:
            super().__init__(4)

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(discard_unknown(knowledge_map.getProbabilityMatrix(knowledge_map.getPlayerName())))


class SurePlaySensor(GenericSensor):
    def __init__(self, n_player: int):
        if n_player == 2 or n_player == 3:
            super().__init__(5)
        else:
            super().__init__(4)

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(play_known(knowledge_map.getProbabilityMatrix(knowledge_map.getPlayerName()),
                                   knowledge_map.getTableCards()))


class RiskyPlaySensor(GenericSensor):
    def __init__(self, n_player: int, probability=0.8):
        if n_player == 2 or n_player == 3:
            super().__init__(5)
        else:
            super().__init__(4)
        self.probability = probability

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(play_unknown(knowledge_map.getProbabilityMatrix(knowledge_map.getPlayerName()), self.probability))


class NoHintLeftSensor(GenericSensor):
    def __init__(self):
        super().__init__(1)

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(knowledge_map.getNoteTokens() == 8)


class HintNumberToPlaySensor(GenericSensor):
    def __init__(self, n_player: int):
        if n_player == 2 or n_player == 3:
            super().__init__(5*(n_player-1))
        else:
            super().__init__(4*(n_player-1))

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(hint_number(knowledge_map))


class HintColorToPlaySensor(GenericSensor):
    def __init__(self, n_player: int):
        if n_player == 2 or n_player == 3:
            super().__init__(5*(n_player-1))
        else:
            super().__init__(4*(n_player-1))

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(hint_color(knowledge_map))


class HintToDiscardSensor(GenericSensor):
    def __init__(self, n_player: int):
        if n_player == 2 or n_player == 3:
            super().__init__(10*(n_player-1))
        else:
            super().__init__(8*(n_player-1))

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(hint_discard(knowledge_map))


def package_sensors(n_player: int):
    return [DiscardKnowSensor(n_player),
            DiscardUnknownSensor(n_player),
            SurePlaySensor(n_player),
            RiskyPlaySensor(n_player),
            NoHintLeftSensor(),
            HintNumberToPlaySensor(n_player),
            HintColorToPlaySensor(n_player),
            HintToDiscardSensor(n_player)]

def get_sensor_len(n_player: int):
    return sum(map(lambda x: x.get_out_size(), package_sensors(n_player)))


def __evaluate_card(probability: ArrayLike, table_cards: Dict[str, List]) -> int:
    """
    return -1 if a card is discardable, 1 if it is playable and 0 if not have enough information
    @param probability: matrix 5x5
    @return: integer
    """
    p = []
    for color in table_cards.keys():
        index = Color.fromstr(color)
        if len(table_cards[color]) == 0:
            p.append(probability[index.value][0])
        elif 0 < len(table_cards[color]) < 5:
            p.append(probability[index.value][table_cards[color][-1].value])
        for card in table_cards[color]:
            if probability[index.value][card.value] == 1:
                # discard
                return -1
    if any([x >= 1 for x in p]):
        # playable
        return 1
    # not enough information on the card
    return 0


def discard_known(my_hand: List[ArrayLike], table_cards: Dict[str, List]) -> List[bool]:
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


def discard_unknown(my_hand: List[ArrayLike]) -> List[bool]:
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


def play_known(my_hand: List[ArrayLike], table_cards: Dict[str, List]) -> List[bool]:
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


def play_unknown(my_hand: List[ArrayLike], prob: float):
    """
    return the bitstring that represent which card of the hand are possible play
    @param my_hand: list of matrix 5x5
    @param prob: float representing the probability accepted
    @return list of booleans
    """
    ret = []
    for i, my_knowledge_matrix in enumerate(my_hand):
        if np.any(np.sum(my_knowledge_matrix, axis=1)[i] >= prob):
            ret.append(True)
        else:
            ret.append(False)
    return ret


def __can_be_played(card, table_cards: Dict[str, List]) -> bool:
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


def __hint_type(knowledge: ArrayLike, card) -> int:
    """
    return information about card on the hand of a player
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





def hint_number(knowledge_map: KnowledgeMap) -> List[bool]:
    """
    return the bitstring that represent which number are hintable
    @param knowledge_map: KnowledgeMap
    @return list of booleans
    """

    def check_number(col: int, hand: List, table_cards: Dict[str, List]):
        ret_val = False
        for card in hand:
            if card.value == col and __can_be_played(card, table_cards):
                ret_val = True
            if card.value == col and not __can_be_played(card, table_cards):
                return False
        return ret_val

    ret = []
    for player in knowledge_map.getPlayerList():
        if player == knowledge_map.getPlayerName():
            continue
        for i in range(1, 6):
            val = check_number(i, knowledge_map.hands[player], knowledge_map.getTableCards())
            ret.append(val)

    return ret


def hint_color(knowledge_map: KnowledgeMap):
    """
    return the bitstring that represent which color are hintable
    @param knowledge_map: KnowledgeMap
    @return list of booleans
    """

    def check_color(col: str, hand: List, table_cards: Dict[str, List]):
        ret_val = False
        for card in hand:
            if card.color == col and __can_be_played(card, table_cards):
                ret_val = True
            if card.color == col and not __can_be_played(card, table_cards):
                return False
        return ret_val

    ret = []
    for player in knowledge_map.getPlayerList():
        if player == knowledge_map.getPlayerName():
            continue
        for color in Color:
            if color == Color.UNKNOWN:
                continue
            val = check_color(Color.fromint(color.value), knowledge_map.hands[player], knowledge_map.getTableCards())
            ret.append(val)

    return ret


def hint_discard(knowledge_map: KnowledgeMap):
    """
    return the bitstring that represent which card are hintable to discard
    @param knowledge_map: KnowledgeMap
    @return list of booleans
    """

    def number_can_be_discarded(value: int, table_cards: Dict[str, List]) -> bool:
        for cards_in_table in table_cards.values():
            if value > len(cards_in_table):
                return False
        return True

    def color_can_be_discarded(c: str, table_cards: Dict[str, List]) -> bool:
        return len(table_cards[c]) == 5

    ret = [False] * 10 * (len(knowledge_map.getPlayerList()) - 1)
    j = 0
    for player in knowledge_map.getPlayerList():
        if player == knowledge_map.getPlayerName():
            continue
        for card in knowledge_map.hands[player]:
            ret[j + card.value - 1] = number_can_be_discarded(card.value, knowledge_map.getTableCards())
            ret[j + 5 + Color.fromstr(card.color).value] = color_can_be_discarded(card.color, knowledge_map.getTableCards())
        j += 10

    return ret
