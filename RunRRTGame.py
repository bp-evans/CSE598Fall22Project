from RRTAgent import RRTAgent
from ObstacleGame import ObstacleGame

def main():
    # Run the game with an RRt agent
    rrt_agent = RRTAgent()
    game = ObstacleGame(rrt_agent)
    game.play(True)


if __name__=="__main__":
    main()