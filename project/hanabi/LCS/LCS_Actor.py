from typing import List
from numpy.typing import NDArray
from hanabi.GameAdapter import GameAdapter


class LCSActor:

    def __init__(self, io: GameAdapter):
        self.io = io

    def act(self, act_string: NDArray) -> None:
        pass

