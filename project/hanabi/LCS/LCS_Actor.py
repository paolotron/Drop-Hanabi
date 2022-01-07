from typing import List
import numpy as np
from hanabi.GameAdapter import GameAdapter, HintType
from numpy.typing import NDArray
from random import choice


class LCSActor:

    def __init__(self, io: GameAdapter):
        self.io = io

    def _get_hint_params(self):
        player = choice(self.io.get_other_players())
        type_hint = choice([HintType.COLOR, HintType.NUMBER])
        value = choice([0, 1, 2, 3, 4]) if type_hint == HintType.NUMBER else choice(
            ['red', 'blue', 'yellow', 'white', 'green'])
        return player, type_hint, value

    def act(self, act_string: NDArray) -> None:
        """
        Calls GameAdapter's functions
        @param act_string: np.ndarray[bool] len = 2
        """
        array = np.array(act_string, dtype=int)

        if len(array) != 2:
            print('WARNING: array length is not 2')

        index = 0
        pow = 0
        for num in array:
            index += num * (2 ^ pow)
            pow += 1

        if index == 3:
            index = np.randint(0, 3)

        if index == 0:
            self.io.send_hint(self._get_hint_params())
        elif index == 1:
            self.io.send_play_card(choice([0, 1, 2, 3, 4]))
        else:
            self.io.send_discard_card(choice([0, 1, 2, 3, 4]))
