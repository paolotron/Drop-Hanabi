from abc import ABC, abstractmethod
from typing import Union
from numpy.typing import NDArray
from numpy import bool_
import numpy as np
import LCS.rules as rules
from knowledge import KnowledgeMap


class GenericSensor(ABC):
    """
    Abstract Class that must be inherited for creating a new sensor
    """
    out_size = 1

    def __init__(self, out_size: int, player: Union[str, None]):
        """
        Constructor for Generic Sensor
        @param out_size: length of the outputted bit_string
        """
        player = player
        out_size = out_size

    @abstractmethod
    def activate(self, knowledge_map: KnowledgeMap) -> NDArray[bool_]:
        """
        Activate method that must be overloaded
        @param knowledge_map: Environment descriptor
        @return: bool array of 'self.out_size' length
        """
        pass


class DiscardKnowSensor(GenericSensor):
    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(rules.discard_known(knowledge_map.getOnePlayerHand(knowledge_map.getPlayerName()),
                                            knowledge_map.getTableCards()))


class DiscardUnknownSensor(GenericSensor):
    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(rules.discard_unknown(knowledge_map.getOnePlayerHand(knowledge_map.getPlayerName())))


class SurePlaySensor(GenericSensor):
    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(rules.play_known(knowledge_map.getOnePlayerHand(knowledge_map.getPlayerName()),
                                         knowledge_map.getTableCards()))


class RiskyPlaySensor(GenericSensor):
    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(rules.play_unknown(knowledge_map.getOnePlayerHand(knowledge_map.getPlayerName())))


class NoHintLeftSensor(GenericSensor):
    def activate(self, knowledge_map) -> NDArray[bool_]:
        return knowledge_map.getNoteTokens() == 8


class HintNumberToPlaySensor(GenericSensor):
    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(rules.hint_number(knowledge_map))


class HintColorToPlaySensor(GenericSensor):
    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(rules.hint_color(knowledge_map))


class HintToDiscardSensor(GenericSensor):
    def activate(self, knowledge_map) -> NDArray[bool_]:
        return np.array(rules.hint_discard(knowledge_map))

