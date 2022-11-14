import argparse
from BCRRT import BCRRTAgent, ImageBCRRRTAgent
from RRTAgent import RRTAgent
from ObstacleGame import ObstacleGame
import DeveloperName


def main(parsed_args):
    # Run the game with an RRT agent
    if not parsed_args.behavior_cloning:
        print("Running game with an RRT Agent")
        rrt_agent = RRTAgent()
    else:
        if parsed_args.image_model:
            print("Running game with image CNN model " + str(parsed_args.model))
            rrt_agent = ImageBCRRRTAgent(parsed_args.model)
        else:
            print("Running game with a BC RRT agent using " + str(parsed_args.model) + " model.")
            rrt_agent = BCRRTAgent(parsed_args.model)
    game = ObstacleGame(rrt_agent)
    game.play(visual=True)


if __name__ == "__main__":
    generate_obstacles = False
    parser = argparse.ArgumentParser(
        prog='Start RRT',
        description='Run the game and traverse')

    parser.add_argument('-bc', '--behavior_cloning', action='store_true')
    parser.add_argument('-img', '--image_model', help="Whether to run in image JIT model mode or not", action='store_true')
    parser.add_argument('-m', '--model', type=str, help="The model file name to use.", default=DeveloperName.my_name + 'model.pk')
    args = parser.parse_args()
    main(args)
