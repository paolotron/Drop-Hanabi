import socket
import time
from abc import ABC, abstractmethod
from typing import Union, List
from constants import *

import GameData
from enum import Enum

from knowledge import KnowledgeMap


class HintType(Enum):
    NUMBER = 0
    COLOR = 1


# Debug Variables
verbose = False
verbose_min = False
verbose_game = False


class EndGameException(Exception):
    pass


# noinspection PyTypeChecker
class GameAdapter:
    """
    Class to play hanabi with a server.
    Use it in a for loop to iterate through the game
    """

    def __init__(self, name: str, *, ip: str = '127.0.0.1', port: int = 1026, datasize: int = 10240):
        """
        Initialize Game Manager creating a connection with the server
        @param name: Player Name
        @param ip: Host IP
        @param port: Process Port
        @param datasize: Size of the socket packets
        """
        self.name = name
        self.ip = ip
        self.port = port
        self.datasize = datasize
        self.move_history = []
        self.action = None
        self.game_end = False
        self.board_state = None
        self.board_state: GameData.ServerGameStateData
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.final_score = None
        while True:
            try:
                self.socket.connect((ip, port))
                break
            except ConnectionRefusedError:
                time.sleep(0.001)
        self.socket.send(GameData.ClientPlayerAddData(name).serialize())
        assert type(GameData.GameData.deserialize(self.socket.recv(datasize))) is GameData.ServerPlayerConnectionOk
        if verbose:
            print("Connection accepted by the server. Welcome " + name)
            print("[" + name + " - " + "Lobby" + "]: ", end="")
        self.socket.send(GameData.ClientPlayerStartRequest(name).serialize())
        data = GameData.GameData.deserialize(self.socket.recv(datasize))
        assert type(data) is GameData.ServerPlayerStartRequestAccepted
        data = GameData.GameData.deserialize(self.socket.recv(datasize))
        assert type(data) is GameData.ServerStartGameData
        self.socket.send(GameData.ClientPlayerReadyData(name).serialize())
        self.players = tuple(data.players)
        self.knowledgeMap = KnowledgeMap(list(self.players), self.name)
        time.sleep(0.01)

    def _request_state(self) -> GameData.ServerGameStateData:
        """
        Request Board State
        """
        data = GameData.ClientGetGameStateRequest(self.name)
        self.socket.send(data.serialize())
        if verbose_min:
            print(f"{self.name}: OPEN")
        while True:

            request = GameData.GameData.deserialize(self.socket.recv(self.datasize))
            t = self._register_action(request)
            if type(t) is GameData.ServerGameStateData or self.game_end:
                if verbose_min:
                    print(f"{self.name}: CLOSE {type(request)} : {self.board_state.currentPlayer}")
                break

    def __iter__(self):
        """
        create iterator for proceeding through the game
        @return: self
        """
        self.current = 0
        return self

    def __next__(self) -> KnowledgeMap:
        """
        next step in the iteration
        returns the current state of the board and the list of all moves
        @rtype: board_state, move_list
        """
        if self.game_end:
            raise StopIteration

        try:
            self._request_state()
        except (ConnectionResetError, EndGameException) as e:
            raise StopIteration
        while self.board_state.currentPlayer != self.name:
            try:
                if self.game_end:
                    raise StopIteration
                response = GameData.GameData.deserialize(self.socket.recv(self.datasize))
                self._register_action(response)
                self._request_state()
            except (ConnectionResetError, EndGameException) as e:
                raise StopIteration
        self.knowledgeMap.updateHands(self.move_history, self.board_state)
        return self.knowledgeMap

    def _send_action(self, action: GameData.ClientToServerData):
        """
        send action to the socket
        @param action: GameData
        """

        self.socket.send(action.serialize())

    def _register_action(self, response: GameData.ServerToClientData):
        if verbose:
            print(f"{self.name}: RECIEVED {response}")
        if type(response) is GameData.ServerGameStateData:
            response: GameData.ServerGameStateData
            self.board_state = response
        elif type(response) is GameData.ServerHintData:
            response: GameData.ServerHintData
            self.move_history.append(response)
        elif type(response) is GameData.ServerPlayerMoveOk:
            response: GameData.ServerPlayerMoveOk
            self.move_history.append(response)
        elif type(response) is GameData.ServerPlayerThunderStrike:
            response: GameData.ServerPlayerThunderStrike
            self.move_history.append(response)
        elif type(response) is GameData.ServerActionValid:
            response: GameData.ServerActionValid
            self.move_history.append(response)
        elif type(response) is GameData.ServerGameOver:
            response: GameData.ServerGameOver
            self.game_end = True
            self.final_score = response.score
            raise EndGameException()
        elif type(response) is GameData.ServerInvalidDataReceived and verbose:
            print(f"{self.name} TURN: {type(response)}, data: {response.data}")
        return response

    def reset(self):
        self.move_history = []
        self.action = None
        self.game_end = False
        self.board_state = None
        self.knowledgeMap = KnowledgeMap(list(self.players), self.name)

    def get_all_players(self):
        """
        Get all players
        @return: tuple(str)
        """
        return list(self.players)

    def get_other_players(self):
        """
        Get all players but the playing one
        @return: tuple(str)
        """
        p = list(self.players)
        p.remove(self.name)
        return tuple(p)

    def _wait_for(self, message_types: List[GameData.GameData]):
        """
        Wait for a specific list of responses
        @param message_types:
        @return: response
        """
        response = GameData.GameData.deserialize(self.socket.recv(self.datasize))
        while type(response) not in message_types and not self.game_end:
            if verbose:
                print(f"{self.name} WAITING FOR {message_types}, RECIEVED {type(response)}")
            self._register_action(response)
            response = GameData.GameData.deserialize(self.socket.recv(self.datasize))
        self._register_action(response)
        return response

    def send_hint(self, player: Union[str, int], type_h: HintType, val: Union[str, int]) -> bool:
        """
        Send a hint to a specific player
        @param player: player receiving the hint
        @param type_h: type of the hint to be sent
        @param val: value or colour
        @return: True if the hint was sent successfully
        """
        type_h = {HintType.NUMBER: 'value', HintType.COLOR: 'colour'}[type_h]
        if verbose_game:
            print(f"{self.name} SENDING HINT TO {player} : {type_h}, {val}")
        try:
            self._send_action(GameData.ClientHintData(self.name, player, type_h, val))
            result = self._wait_for(
                [GameData.ServerActionInvalid, GameData.ServerInvalidDataReceived, GameData.ServerHintData])
        except (ConnectionResetError, EndGameException) as e:
            return True

        if type(result) in [GameData.ServerActionInvalid, GameData.ServerInvalidDataReceived, GameData.ServerGameOver]:
            return False
        if type(result) in [GameData.ServerHintData, GameData.ServerGameOver]:
            return True
        raise ValueError

    def send_play_card(self, card_number: int) -> bool:
        """
        Play a card from hand
        @param card_number: index of the card
        @return: True if the card was correct False otherwise
        """
        try:
            self._send_action(GameData.ClientPlayerPlayCardRequest(self.name, card_number))
            result = self._wait_for([GameData.ServerActionInvalid,
                                     GameData.ServerInvalidDataReceived,
                                     GameData.ServerPlayerThunderStrike,
                                     GameData.ServerPlayerMoveOk])

        except (ConnectionResetError, EndGameException) as _:
            return True
        if type(result) is GameData.ServerPlayerMoveOk and verbose_game:
            print(f"PLAYED {result.card} FROM {self.name} OK!")
        if type(result) is GameData.ServerPlayerThunderStrike and verbose_game:
            print(f"PLAYED {result.card} FROM {self.name} THUNDERSTRIKE!")
        if type(result) in [GameData.ServerPlayerMoveOk, GameData.ServerPlayerThunderStrike, GameData.ServerGameOver]:
            return True
        if type(result) in [GameData.ServerInvalidDataReceived, GameData.ServerActionInvalid]:
            return False
        raise ValueError

    def send_discard_card(self, card_number: int) -> bool:
        """
        Discard a card
        @param card_number: card index
        @return: if the card was successfully discarded
        """
        try:
            self._send_action(GameData.ClientPlayerDiscardCardRequest(self.name, card_number))
            result = self._wait_for([GameData.ServerActionValid, GameData.ServerActionInvalid])
        except (ConnectionResetError, EndGameException) as e:
            return True
        if verbose_game and type(result) is GameData.ServerActionValid:
            print(f"DISCARDED {result.card} FROM {self.name}")
        if type(result) in [GameData.ServerActionValid, GameData.ServerGameOver]:
            return True
        if type(result) is GameData.ServerActionInvalid:
            return False
        raise ValueError

    def end_game_data(self):
        return {
            "n_turns": len(self.move_history),
            "points": self.final_score,
            "loss": self.final_score == 0
        }


class Player(ABC):
    """
    Abstract class for a generic player
    """

    def __init__(self, name: str, conn_params=None):
        """
        Init method for the generic player, set ups the connection parameters
        @param name: str
        @param conn_params: dictionary with name, ip, port, datasize
        """
        self.name = name
        if conn_params is None:
            self.start_dict = {
                'name': name,
                'ip': HOST,
                'port': PORT,
                'datasize': DATASIZE,
            }
        else:
            self.start_dict = conn_params
        self.io: GameAdapter
        self.io = None

    @abstractmethod
    def make_action(self, state: KnowledgeMap):
        """
        Method for making an action given the current state
        @param state: knowledge map representing the state
        @return: None
        """
        pass

    def setup(self, *args, **kwargs):
        """
        Generic setup method, called before starting the game, can be inherithed
        @param args:
        @param kwargs:
        @return:
        """
        pass

    def cleanup(self):
        """
        Generic cleanup method, called after finishing the game, can be inherithed
        @return:
        """
        pass

    def end_game_data(self):
        """
        return the result of the match
        @return: dict with end game info
        """
        return self.io.end_game_data()

    def start(self, *args, **kwargs):
        """
        method to call for starting the game and running it
        runs setup -> make_action -> ... -> make_action -> cleanup
        @param args:
        @param kwargs:
        @return: None
        """

        if self.io is None:
            self.io = GameAdapter(**self.start_dict)
        else:
            self.io.reset()

        self.setup(*args, **kwargs)
        for state in self.io:
            self.make_action(state)

        self.cleanup()
