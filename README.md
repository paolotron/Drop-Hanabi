# Drop - Hanabi
Drop-Hanabi is the result of the hanabi project from the Computational Intelligence Course @ Polito.
The project implements an hanabi player based on a **Learning Classifier System**.
## Usage
To start a player:
1. Start server.py
```
python project/hanabi/server.py n_players
```
2. Start the player

```
python project/hanabi/player_LCS.py name n_players
```
3. Start other players from player_LCS or other scripts
4. When the specified number of players is reached the server will start automatically the match

take care in specifying the correct number of players for both the clients and the server.
