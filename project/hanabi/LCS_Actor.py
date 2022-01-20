import numpy as np
from numpy.typing import ArrayLike

from GameAdapter import GameAdapter, HintType


class LCSActor:
    """
    Given a 6-bits act string, the actor make the action, calling the right function with the right parameters
    """

    def __init__(self, io: GameAdapter):
        """
        paramDict: Dictionary containing all the parameters to use in the actions, for each bitstring
        """
        self.io = io
        self.action_length = 6
        self.paramDict = {}
        self.players = self.io.get_other_players()
        self.num_cards = 5 if len(self.io.get_all_players()) < 4 else 4
        self.max_index = self.num_cards * 2 + 10 * len(self.players)

        for i in range(self.num_cards):
            self.paramDict[i] = i
            self.paramDict[i + self.num_cards] = i

        for i, player in enumerate(self.players):
            self.paramDict[self.num_cards * 2 + i * 10 + 0] = (player, HintType.NUMBER, 1)
            self.paramDict[self.num_cards * 2 + i * 10 + 1] = (player, HintType.NUMBER, 2)
            self.paramDict[self.num_cards * 2 + i * 10 + 2] = (player, HintType.NUMBER, 3)
            self.paramDict[self.num_cards * 2 + i * 10 + 3] = (player, HintType.NUMBER, 4)
            self.paramDict[self.num_cards * 2 + i * 10 + 4] = (player, HintType.NUMBER, 5)
            self.paramDict[self.num_cards * 2 + i * 10 + 5] = (player, HintType.COLOR, "red")
            self.paramDict[self.num_cards * 2 + i * 10 + 6] = (player, HintType.COLOR, "blue")
            self.paramDict[self.num_cards * 2 + i * 10 + 7] = (player, HintType.COLOR, "green")
            self.paramDict[self.num_cards * 2 + i * 10 + 8] = (player, HintType.COLOR, "yellow")
            self.paramDict[self.num_cards * 2 + i * 10 + 9] = (player, HintType.COLOR, "white")

    @staticmethod
    def get_action_length(num_players: int):
        # Coincidence that for all possible number of players we always need 6 bits
        return 6

    def act(self, act_string: ArrayLike) -> bool:
        """
        Calls GameAdapter's functions, based on the act_string
        @param act_string: np.ndarray[bool] len = 6
        @return True or False
        """
        act_array = np.array(act_string, dtype=int)

        if len(act_array) != 6:
            print('WARNING: array length is not 6')

        index = int(''.join(map(str, act_array))[::-1], base=2)

        index = index % self.max_index
        if 0 <= index < self.num_cards:
            res = self.io.send_play_card(self.paramDict[index])
        elif self.num_cards <= index < self.num_cards * 2:
            res = self.io.send_discard_card(self.paramDict[index])
        else:
            res = self.io.send_hint(*self.paramDict[index])
        return res


