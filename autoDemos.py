import argparse
from BCRRT import BCRRTAgent
from RRTAgent import RRTAgent
from ObstacleGame import ObstacleGame
import time
import uuid
import os
import pandas as pd
import pygame
import random
from Agents import Agent
from RRTAgent import Observer
from Configuration import StaticObstaclesConfiguration, DynamicObstaclesConfiguration
from RRTAbstract import RRT_Core, RRTObserver
import numpy as np
from randomObs import setValsRandom

def main(parsed_args):
    num_demos = int(parsed_args.n)
    dimensions = (500, 800)
    maph, mapw = dimensions

    StaticObstaclesConfiguration.mapw = mapw
    StaticObstaclesConfiguration.maph = maph

    display_map = pygame.display.set_mode((mapw, maph))

    goal = (800, 300) # change this later

    if(int(parsed_args.d) == 1):
        print("Dynamic Mode")
        goalx = random.randint(200,800)
        goaly = random.randint(0,500)
        goal = (goalx, goaly)
        setValsRandom(goal) # set goal for randomObs
        conf = DynamicObstaclesConfiguration((50, 50), goal)
    else:
        print("Static Mode")
        conf = StaticObstaclesConfiguration((50, 50), goal)

    # Grab current demonstration labels
    demonstration_label_file = "ImageLabels.csv"
    images_dir = "image_demos/"
    try:
        demonstration_labels = pd.read_csv(demonstration_label_file)
    except FileNotFoundError:
        # File doesn't exist, create a new dataset
        demonstration_labels = pd.DataFrame({"Image Name": [], "Label": []})

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Run RRT

    for i in range(0,num_demos):
        print("Running RRT")
        rrt = RRT_Core([conf])
        graph, path = rrt.RRTAlg(10000, Observer(None, conf)) # change None to displayMap if you want to visualize the RRT
    
        # Testing path contents
        print("Path Length:")
        print(len(path))
        print("")

        # Graphics loop
        print("Visualizing game state")
        # Loops through path to goal showing agent position and saving screenshots with action information
        for x in path:
            vec = x[0].as_vector()
            if(int(parsed_args.d) == 1):
                new_conf = DynamicObstaclesConfiguration((vec[0], vec[1]), goal)
            else:
                new_conf = StaticObstaclesConfiguration((vec[0], vec[1]), goal)

            new_conf.visualize(display_map)
            pygame.event.pump()
            filename = str(uuid.uuid1()) + ".jpg"
            pygame.image.save(display_map, images_dir+filename)
            # Save this label
            if (not (x[1] == None)):
                new_label = {"Image Name": [filename], "Label": [x[1].value]}
            # demonstration_labels.iloc[len(demonstration_labels.index)] = [filename, action]
            demonstration_labels = pd.concat([demonstration_labels, pd.DataFrame(new_label)])
            # demonstration_labels.append(new_label, ignore_index=True)
            time.sleep(.05)

    print("Demos Ended")
    
    # Save demonstrations
    demonstration_labels.to_csv(demonstration_label_file)






if __name__ == "__main__":
    generate_obstacles = False
    parser = argparse.ArgumentParser(
        prog='RRT Demos',
        description='Run auto generated RRT demos')

    parser.add_argument('-n')
    parser.add_argument('-d')
    args = parser.parse_args()
    main(args)
