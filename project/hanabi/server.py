import os
import GameData as GameData
import socket
from game import Game
import threading
import logging
import sys

from constants import *


def manageConnection(conn: socket, addr):
    global status
    with conn:
        logging.info("Connected by: " + str(addr))
        keepActive = True
        playerName = ""
        while keepActive:
            data = conn.recv(DATASIZE)
            if not data:
                del playerConnections[playerName]
                logging.warning("Player disconnected: " + playerName)
                game.removePlayer(playerName)
                keepActive = False
            else:
                data = GameData.GameData.deserialize(data)
                if status == "Lobby":
                    if type(data) is GameData.ClientPlayerAddData:
                        playerName = data.sender
                        commandQueue[playerName] = []
                        playerConnections[playerName] = (conn, addr)
                        logging.info("Player connected: " + playerName)
                        game.addPlayer(playerName)
                        conn.send(GameData.ServerPlayerConnectionOk(playerName).serialize())
                    elif type(data) is GameData.ClientPlayerStartRequest:
                        if playerName not in game.getPlayers() and playerName != "" and playerName is not None:
                            game.setPlayerReady(playerName)
                            logging.info("Player ready: " + playerName)
                            conn.send(GameData.ServerPlayerStartRequestAccepted(len(game.getPlayers()),
                                                                                game.getNumReadyPlayers()).serialize())
                        else:
                            return
                        if len(game.getPlayers()) == game.getNumReadyPlayers() and len(game.getPlayers()) > 1:
                            listNames = []
                            for player in game.getPlayers():
                                listNames.append(player.name)
                            logging.info("Game start! Between: " + str(listNames))
                            for player in playerConnections:
                                playerConnections[player][0].send(GameData.ServerStartGameData(listNames).serialize())
                            game.start()
                    # This ensures every player is ready to send requests
                    elif type(data) is GameData.ClientPlayerReadyData:
                        playersOk.append(1)
                    # If every player is ready to send requests, then the game can start
                    if len(playersOk) == len(game.getPlayers()):
                        status = "Game"
                        for player in commandQueue:
                            for cmd in commandQueue[player]:
                                singleData, multipleData = game.satisfyRequest(cmd, player)
                                if singleData is not None:
                                    playerConnections[player][0].send(singleData.serialize())
                                if multipleData is not None:
                                    for id in playerConnections:
                                        playerConnections[id][0].send(multipleData.serialize())
                                        if game.isGameOver():
                                            os._exit(0)
                        commandQueue.clear()
                    elif type(data) is not GameData.ClientPlayerAddData and type(
                            data) is not GameData.ClientPlayerStartRequest and type(
                        data) is not GameData.ClientPlayerReadyData:
                        commandQueue[playerName].append(data)
                # In game
                elif status == "Game":
                    singleData, multipleData = game.satisfyRequest(data, playerName)
                    if singleData is not None:
                        conn.send(singleData.serialize())
                    if multipleData is not None:
                        for id in playerConnections:
                            playerConnections[id][0].send(multipleData.serialize())
                            if game.isGameOver():
                                os._exit(0)


def manageInput():
    while True:
        data = input()
        if data == "exit":
            logging.info("Closing the server...")
            os._exit(0)


print("Type 'exit' to end the program")


def manageNetwork():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        logging.info("Hanabi server started on " + HOST + ":" + str(PORT))
        while True:
            s.listen()
            conn, addr = s.accept()
            threading.Thread(target=manageConnection, args=(conn, addr)).start()


# SERVER
playerConnections = {}
game = Game()

playersOk = []

statuses = [
    "Lobby",
    "Game"
]
status = statuses[0]

commandQueue = {}


def start_server():
    logging.basicConfig(filename="game.log", level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt="%m/%d/%Y %I:%M:%S %p")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    threading.Thread(target=manageNetwork).start()
    manageInput()


if __name__ == '__main__':
    start_server()
