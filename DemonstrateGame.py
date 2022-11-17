"""
This file can be run to let an expert play the game and will save the experts demonstrations
"""
from ObstacleGame import ObstacleGame
import argparse
import csv
import keyboardAgents
import os
import DeveloperName


def main(parsed_args):
    # def save_demo(demo, file_path='demo.csv'):
    #     """
    #     Write demonstration into demo csv file.
    #
    #     Args:
    #         demonstration (list[Configuration, Action]): list of tuples of config state and action.
    #         file_path (str): Path to the file to write demonstration dataset into.
    #
    #     :return:
    #     """
    #     write_header = True
    #     csv_header = ["config", "action"]
    #     if os.path.exists(file_path):
    #         write_header = False
    #     with open(file_path, 'a', newline='') as csv_file_descriptor:
    #         csv_writer = csv.writer(csv_file_descriptor)
    #         if write_header:
    #             csv_writer.writerow(csv_header)
    #         for config, action in demo:
    #             csv_writer.writerow([config.as_vector(), action.value])

    # Start up the game
    keyboard_agent = keyboardAgents.KeyboardAgent()
    game = ObstacleGame(keyboard_agent, parsed_args.collect)

    print("Play the game")
    demo = game.play(isDynamic=parsed_args.dynamic, visual=True)


if __name__ == "__main__":
    generate_obstacles = False
    parser = argparse.ArgumentParser(
        prog='Run demonstration game',
        description='Run the game to collect demonstration using [WASD]')

    parser.add_argument('-d', '--dynamic', action='store_true')
    parser.add_argument('-c', '--collect', action='store_true')
    args = parser.parse_args()
    main(args)
