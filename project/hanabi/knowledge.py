from enum import Enum
import numpy as np


class Color(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3
    WHITE = 4
    UNKNOWN = 5

    @staticmethod
    def fromstr(string: str):
        string = string.lower()
        dic = {"red": Color.RED, "blue": Color.BLUE, "green": Color.GREEN, "yellow": Color.YELLOW, "white": Color.WHITE}
        return dic[string]

    @staticmethod
    def fromint(number: int):
        dic = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "white"}
        return dic[number]


class KnowledgeMap:
    """
    Contains information about other players hands
    WARNING: this class does not know if the players have less cards in the hand
    than the maximum number of holdable cards(this happens only in end game)
    More information on getPlayerHand description
    """

    def __init__(self, players, name):
        """
        @param players: list of all players, obtained with GameAdapter.get_all_players()
        @param name: str, name of the playing player

        name = name of the main player
        players = list of strings with names of ALL players
        numPlayers = number of players in the game
        numCards = number of cards for each player
        hands = dictionary, key = name of a player, value = player's hand
        hints = dictionary, key = nema of a player, value = boolean matrix, True if it's possible card False otherwise (initially all True)
        tableCards = list of cards, containing only the highest value played car for each color
        discardPile = list of discarded cards
        usedNoteTokens = int
        usedStormTokens = int
        deckCards = int, number of cards remaining in the deck

        """
        self.name = name
        self.players = players
        self.numPlayers = len(players)
        self.numCards = 4 if self.numPlayers > 3 else 5
        self.matrix = np.array([[3, 2, 2, 2, 1],
                                [3, 2, 2, 2, 1],
                                [3, 2, 2, 2, 1],
                                [3, 2, 2, 2, 1],
                                [3, 2, 2, 2, 1]])
        self.numMoves = 0
        self.tableCards = {}
        self.discardPile = []
        self.usedNoteTokens = 0
        self.usedStormTokens = 0
        self.deckCards = 50 - self.numPlayers * self.numCards
        self.hands = {}
        self.hints = {}
        for player in players:
            self.hints[player] = []
            for _ in range(self.numCards):
                self.hints[player].append(np.ones((5, 5), dtype=bool))

    def __updateHint(self, player, move):
        """
        updates self.hints with the given hint
        for each matrix in the dict (which represents a single card),
        puts to False all the positions that are excluded as possible cards
        @param player: hinted player's name
        @param move: move taken from GameAdapter's move_history, only hints
        """
        for i, card in enumerate(self.hints[player]):
            val = move.value - 1 if move.type == 'value' else Color.fromstr(move.value).value
            if i in move.positions:
                if move.type == 'value':
                    card[:, :val] = False
                    if val < self.numCards - 1:
                        card[:, val + 1:] = False
                else:
                    card[:val, :] = False
                    if val < self.numCards - 1:
                        card[val + 1:, :] = False
            else:
                if move.type == 'value':
                    card[:, val] = False
                else:
                    card[val, :] = False

    def __updateMatrix(self, move):
        """
        updates self.matrix by subtracting the card played or discarded from the matrix
        removes one matrix in self.hints at the index corresponding to the index of the card played or discarderd
        adds a new matrix at the end of the hints, representing the drawn card
        @param move: move taken from GameAdapter's move_history, only play or discard
        """
        self.matrix[Color.fromstr(move.card.color).value, move.card.value - 1] -= 1
        self.hints[move.lastPlayer].pop(move.cardHandIndex)
        if self.deckCards > 0:
            self.hints[move.lastPlayer].append(np.ones((5, 5), dtype=bool))
        else:
            self.hints[move.lastPlayer].append(np.zeros((5, 5), dtype=bool))
        self.deckCards -= 1

    def updateHands(self, move_history, state):
        """
        Updates the state of the game looking at the move_history
        hands, discardPile, tableCards, usedNoteTokens and usedStormTokens are directly taken from the state
        hints and matrix are updated with custom functions
        @param move_history: move_history from GameAdapter
        @param state: board_state from GameAdapter
        """

        self.discardPile = state.discardPile
        self.tableCards = state.tableCards
        self.usedNoteTokens = state.usedNoteTokens
        self.usedStormTokens = state.usedStormTokens
        for i in reversed(range(len(move_history) - self.numMoves + 1)):
            if i > 0:
                moveType = str(type(move_history[-i])).split(".")
                moveType = moveType[len(moveType) - 1][:-2]
                if moveType == "ServerHintData":
                    self.__updateHint(move_history[-i].destination, move_history[-i])
                elif moveType in \
                        ["ServerPlayerMoveOk",
                         "ServerPlayerThunderStrike",
                         "ServerActionValid"]:
                    self.__updateMatrix(move_history[-i])
        for player in state.players:
            self.hands[player.name] = player.hand
        self.numMoves = len(move_history)

    def getProbabilityMatrix(self, target, probability=True):
        """
            Compute probability matrix for each card in the target player's hand
            @param target: the player name (string) you want to inspect
            @param probability: if true returns probabilities, otherwise number of cards (used for tests)
            @return: list(5x5 numpy array)
            In case of less cards in the hand than the maximum number of holdable cards
            the list returned is still as long as the maximum number of cards
            but you should ignore the elements of non-existing cards
        """
        def getProb(mat, h):
            if np.any(h):
                return mat * h / (mat * h).sum()
            else:
                return np.zeros((5, 5))
        tmpMatrix = self.matrix.copy()
        for player in self.players:
            if player != target and player != self.name:
                for card in self.hands[player]:
                    tmpMatrix[Color.fromstr(card.color).value, card.value - 1] -= 1
        if probability:

            return [getProb(tmpMatrix, h) for h in self.hints[target]]
        else:
            return [tmpMatrix * h for h in self.hints[target]]

    def getPlayerName(self):
        return self.name

    def getTableCards(self):
        return self.tableCards

    def getDiscardPile(self):
        return self.discardPile

    def getPlayerList(self):
        """
        @return list(str)
        """
        return self.players

    def getPlayerHands(self):
        """
        @return dict(key=playerName, value=list(Card))
        """
        return self.hands

    def getStormTokens(self):
        return self.usedStormTokens

    def getNoteTokens(self):
        return self.usedNoteTokens

    def getOnePlayerHand(self, target):
        """
        @param target: string with name of the player to inspect
        @return list(Card)
        """
        return self.hands[target]
