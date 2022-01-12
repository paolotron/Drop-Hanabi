import os

import subprocess
import time
from threading import Thread

import player_Paolo
import hanabi.server as server
if __name__ == '__main__':
    server = Thread(target=server.start_server)
    server.start()
    players = [player_Paolo, player_Paolo]
    args = [('Davide',), ('Matteo',)]
    threads = [Thread(target=player.main, args=arg)
               for player, arg in zip(players, args)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    server.join()
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    os.rename('./game.log', f'./logs/{time.strftime("%y%m%d%H%M%S")}.log')
