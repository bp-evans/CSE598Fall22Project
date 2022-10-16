import argparse


from BCRRT import BCRRT
from RRTAgent import RRTAgent, BCRRTAgent
from ObstacleGame import ObstacleGame


def main(parsed_args):
    # Run the game with an RRt agent
    if not parsed_args.behavior_cloning:
        rrt_agent = RRTAgent()
    else:
        preset_conf = ObstacleGame.get_start_goal_preset()
        rrt_agent = BCRRTAgent(BCRRT(preset_conf))
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
