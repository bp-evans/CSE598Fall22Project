"""
This file can be run to let an expert play the game and will save the experts demonstrations
"""
from ObstacleGame import ObstacleGame
import keyboardAgents


def main():
    # Start up the game
    keyboard_agent = keyboardAgents.KeyboardAgent()
    game = ObstacleGame(keyboard_agent)

    print("Play the game")
    game.play(True)

    pass


if __name__ == "__main__":
    main()
