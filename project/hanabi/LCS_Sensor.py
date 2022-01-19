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

    def get_out_size(self) -> int:
        """
        Method for deducing the sensor size
        @return: int
        """
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
        """
        Method for getting the activation of a specific sensor
        @param knowledge_map:
        @return: NDArray
        """
        res = self.activate(knowledge_map)
        assert res.size == self.out_size
        return res


class DiscardKnowSensor(GenericSensor):
    """
    TODO Add Description
    """
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
    """
    TODO Add Description
    """
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
    """
    TODO Add Description
    """
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
    """
    TODO Add Description
    """
    def __init__(self, n_player: int, probability=0.8):
        if n_player == 2 or n_player == 3:
            super().__init__(5)
        else:
            super().__init__(4)
        self.probability = probability

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(play_unknown(knowledge_map.getProbabilityMatrix(knowledge_map.getPlayerName()),
                                     knowledge_map.hints[knowledge_map.getPlayerName()], 0.8))


class NoHintLeftSensor(GenericSensor):
    """
    TODO Add Description
    """
    def __init__(self):
        super().__init__(1)

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(knowledge_map.getNoteTokens() == 8)


class UselessDiscardSensor(GenericSensor):
    """
    TODO Add Description
    """
    def __init__(self):
        super().__init__(1)

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(knowledge_map.getNoteTokens() == 0)


class ThunderStrikeMoment(GenericSensor):
    """
    TODO Add Description
    """
    def __init__(self):
        super().__init__(1)

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        if knowledge_map.getStormTokens() == 2:
            pass
        return np.array(knowledge_map.getStormTokens() == 2)


class HintNumberToPlaySensor(GenericSensor):
    """
    TODO Add Description
    """
    def __init__(self, n_player: int):
        super().__init__(5 * (n_player - 1))

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(hint_number(knowledge_map))


class HintColorToPlaySensor(GenericSensor):
    """
    TODO Add Description
    """
    def __init__(self, n_player: int):
        super().__init__(5 * (n_player - 1))

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(hint_color(knowledge_map))


class HintToDiscardSensor(GenericSensor):
    """
    TODO Add Description
    """
    def __init__(self, n_player: int):
        super().__init__(10 * (n_player - 1))

    def get_out_size(self):
        return super().get_out_size()

    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(hint_discard(knowledge_map))


def get_debug_string(n_players, sensor_type, idx=0):
    """
    TODO DELETE THIS FUNCTION
    """
    i = 0
    for sens in package_sensors(n_players):
        if type(sens) is sensor_type:
            return i + idx
        i += sens.get_out_size()


def package_sensors(n_player: int):
    """
    get default list of all used sensors
    @param n_player: int
    @return: List[GenericSensor]
    """
    return [DiscardKnowSensor(n_player),
            DiscardUnknownSensor(n_player),
            SurePlaySensor(n_player),
            RiskyPlaySensor(n_player),
            NoHintLeftSensor(),
            HintNumberToPlaySensor(n_player),
            HintColorToPlaySensor(n_player),
            HintToDiscardSensor(n_player),
            UselessDiscardSensor(),
            ThunderStrikeMoment()]


def get_sensor_len(n_player: int):
    """
    Get the length of the default sensor list
    @param n_player: int
    @return: int
    """
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
            if probability[index.value][card.value - 1] == 1:
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


def play_unknown(my_hand: List[ArrayLike], hints: List[ArrayLike], prob: float):
    """
    return the bitstring that represent which card of the hand are possible play
    @param my_hand: list of matrix 5x5
    @param prob: float representing the probability accepted
    @param hints: list of matrix 5x5
    @return list of booleans
    """
    ret = check_hint_matrix(hints)
    if not any(ret):
        ret = []
        for i, my_knowledge_matrix in enumerate(my_hand):
            if np.any(np.sum(my_knowledge_matrix, axis=1)[i] >= prob):
                ret.append(True)
            else:
                ret.append(False)
    return ret


def check_hint_matrix(hints: List[ArrayLike]):
    """
    TODO Add Descritption
    @param hints:
    @return:
    """
    r = []
    for card in hints:
        r.append(np.sum(np.any(card, axis=1)) == 1 or np.sum(np.any(card, axis=0)) == 1)
    return r


def play_hinted(hints: List[ArrayLike], num_cards: int, table_cards: Dict[str, List]):
    """
    TODO Add Descritption
    @param hints:
    @param num_cards:
    @param table_cards:
    @return:
    """
    ret = []
    numbers, colors = hints_received(hints, num_cards)
    lengths = [len(table_cards[i]) for i in table_cards.keys()]
    for n, c in zip(numbers, colors):
        if n != -1 and any([ll == n - 1 for ll in lengths]):
            ret.append(True)
        elif c != -1 and len(table_cards[Color.fromint(c)]) != 5:
            ret.append(True)
        else:
            ret.append(False)
    return ret


def hints_received(hints: List[ArrayLike], num_cards: int):
    """
    TODO Add Descritption
    """
    ret_n = [-1 for _ in range(num_cards)]
    ret_c = [-1 for _ in range(num_cards)]
    # ret_n and ret_c will tell, for each card in my hand, if it has been hinted its color
    # its number
    for i in range(num_cards):
        # for each card in my hand check if it has been hinted and if it's playable
        # for each number and for each color
        for j in range(5):
            if np.any(hints[i][:, j]):
                ret_n[i] = j + 1
                break
            elif np.any(hints[i][j, :]):
                ret_c[i] = j
                break
    return ret_n, ret_c


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
    if np.sum(knowledge, axis=1)[index.value] >= 1:
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

    def check_number(col: int, hand: List, table_cards: Dict[str, List], player_hand: List[ArrayLike]):
        ret_val = False
        for card, matrix in zip(hand, player_hand):
            if __hint_type(matrix, card) not in (0, 1):
                if card.value == col and __can_be_played(card, table_cards):
                    ret_val = True
                if card.value == col and not __can_be_played(card, table_cards):
                    return False
        return ret_val

    ret = []
    for player in knowledge_map.getPlayerList():
        hand_player = knowledge_map.getProbabilityMatrix(player)
        if player == knowledge_map.getPlayerName():
            continue
        for i in range(1, 6):
            val = check_number(i,
                               knowledge_map.hands[player],
                               knowledge_map.getTableCards(),
                               hand_player)
            ret.append(val)

    return ret


def hint_color(knowledge_map: KnowledgeMap):
    """
    return the bitstring that represent which color are hintable
    @param knowledge_map: KnowledgeMap
    @return list of booleans
    """

    def check_color(col: str, hand: List, table_cards: Dict[str, List], player_hand: List[ArrayLike]):
        ret_val = False
        for card, matrix in zip(hand, player_hand):
            if __hint_type(matrix, card) not in (0, 2):
                if card.color == col and __can_be_played(card, table_cards):
                    ret_val = True
                if card.color == col and not __can_be_played(card, table_cards):
                    return False
        return ret_val

    ret = []
    for player in knowledge_map.getPlayerList():
        hand_player = knowledge_map.getProbabilityMatrix(player)
        if player == knowledge_map.getPlayerName():
            continue
        for color in Color:
            if color == Color.UNKNOWN:
                continue
            val = check_color(Color.fromint(color.value),
                              knowledge_map.hands[player],
                              knowledge_map.getTableCards(),
                              hand_player)
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
            ret[j + 5 + Color.fromstr(card.color).value] = color_can_be_discarded(card.color,
                                                                                  knowledge_map.getTableCards())
        j += 10

    return ret


