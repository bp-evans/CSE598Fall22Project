import argparse
from BCRRT import BCRRTAgent
from RRTAgent import RRTAgent
from ObstacleGame import ObstacleGame
import DeveloperName


def main(parsed_args):
    # Run the game with an RRT agent
    if not parsed_args.behavior_cloning:
        rrt_agent = RRTAgent()
    else:
        model_name = DeveloperName.my_name + 'model.pk'
        rrt_agent = BCRRTAgent(model_name)
    game = ObstacleGame(rrt_agent)
    game.play(visual=True)


if __name__ == "__main__":
    generate_obstacles = False
    parser = argparse.ArgumentParser(
        prog='Start RRT',
        description='Run the game and traverse')

    parser.add_argument('-bc', '--behavior_cloning', action='store_true')
    args = parser.parse_args()
    main(args)
