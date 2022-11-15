import argparse
from BCRRT import BCRRTAgent
from RRTAgent import RRTAgent
from ObstacleGame import ObstacleGame
import DeveloperName


def main(parsed_args):
    # Run the game with an RRT agent
    if not parsed_args.behavior_cloning:
        print("Running game with an RRT Agent")
        rrt_agent = RRTAgent()
    else:
        print("Running game with a BC RRT agent using " + str(parsed_args.model) + " model.")
        rrt_agent = BCRRTAgent(parsed_args.model)
    game = ObstacleGame(rrt_agent)
    if int(parsed_args.d) == 1:
        print("Dynamic Obs Game")
        game.play(isDynamic = True, visual=True)
    else:
        print("Static Obs Game")
        game.play(isDynamic = False, visual = True)


if __name__ == "__main__":
    generate_obstacles = False
    parser = argparse.ArgumentParser(
        prog='Start RRT',
        description='Run the game and traverse')

    parser.add_argument('-bc', '--behavior_cloning', action='store_true')
    parser.add_argument('-m', '--model', type=str, help="The model file name to use.", default=DeveloperName.my_name + 'model.pk')
    parser.add_argument('-d')
    args = parser.parse_args()
    main(args)
