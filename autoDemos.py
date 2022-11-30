import argparse
import time
import uuid
import os
import csv
import pygame
from Configuration import StaticObstaclesConfiguration, DynamicObstaclesConfiguration
from RRTAbstract import RRT_Core
import multiprocessing as mp
from itertools import chain
from functools import partial
from typing import Dict, Any
from time import sleep


def record_image_for(conf: StaticObstaclesConfiguration, action, images_dir) -> (str, "DiscreteDirectionAction"):
    """
    Records an image for a configuration and action
    :param images_dir:
    :param action:
    :param conf:
    :return: The action that should be taken from this image
    """
    pygame.init()
    pygame.display.set_mode((100, 100))
    surface = pygame.Surface((conf.mapw, conf.maph))
    conf.visualize(surface)
    filename = str(uuid.uuid1()) + ".jpg"
    pygame.image.save(surface, images_dir + filename)
    return filename, action


def run_demo(i, conf_type, parameters: Dict[str, Any]):
    """
    Generates 1 demo and all the data points associated with it
    :return:
    """
    # Generate a random start conf
    start = conf_type.gen_random_conf(set_start=parameters["start"], set_goal=parameters["goal"])
    # Run RRT
    print(f"Running RRT for iter {i}")
    rrt = RRT_Core([start])
    graph, path = rrt.RRTAlg(2000, None,
                             always_return_path=True)  # change None to an observer with a displayMap if you want to visualize the RRT

    # Testing path contents
    print("Path Length:")
    print(len(path), "\n")

    # Generate image for all confs on path
    print(f"Visualizing {len(path)} state(s)")
    labels = []
    for node, action in path:
        labels.append(record_image_for(node, action, parameters["images_dir"]))
    # labels = map(lambda x: record_image_for(x[0], x[1], images_dir), path)
    # labels = worker_pool.starmap(partial(record_image_for, images_dir=images_dir), path)

    return labels


def main(parsed_args):
    num_demos = int(parsed_args.n)
    dimensions = (500, 800)
    maph, mapw = dimensions

    StaticObstaclesConfiguration.mapw = mapw
    StaticObstaclesConfiguration.maph = maph

    # display_map = pygame.display.set_mode((mapw, maph))

    # goal = (800, 300) # change this later

    if parsed_args.d:
        print("Dynamic Mode")
        conf_type = DynamicObstaclesConfiguration
    else:
        print("Static Mode")
        conf_type = StaticObstaclesConfiguration

    # Grab current demonstration labels
    demonstration_label_file = f"{parsed_args.label}ImageLabels.csv"
    images_dir = f"{parsed_args.label}_image_demos/"
    exists = os.path.isfile(demonstration_label_file)
    demonstration_labels = open(demonstration_label_file, 'a')
    label_writer = csv.DictWriter(demonstration_labels, ["Image Name", "Label"])
    if not exists:
        label_writer.writeheader()

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Set up a queue to handle writing outputs, images can be written without conflict by each process
    # However, the label file needs to be synchronized

    # Run all the demos in a worker pool
    start = time.time()
    pool = mp.Pool(processes=None)

    # Wait for pygame to start up so the welcome messages don't drown out the following input requests
    sleep(1)

    use_dynamic_start = input("Use dynamic start? (y/n)") == "y"
    use_dynamic_goal = input("Use dynamic goal? (y/n)") == "y"

    parameters = {
        "images_dir": images_dir,
        "start": None if use_dynamic_start else (50, 50),
        "goal": None if use_dynamic_goal else (700, 200)
    }

    labels = pool.map(partial(run_demo, conf_type=conf_type, parameters=parameters), range(num_demos))

    print("Finished generating images, saving labels")

    i = 0
    for (image, label) in chain.from_iterable(labels):
        if label is not None:
            new_label = {"Image Name": image, "Label": label.value}
            label_writer.writerow(new_label)
            i += 1

    # Save demonstrations
    demonstration_labels.close()

    end = time.time()

    print(f"Saved {i} labels\n\nFinished in {end - start:.2f} seconds")


if __name__ == "__main__":
    generate_obstacles = False
    parser = argparse.ArgumentParser(
        prog='RRT Demos',
        description='Run auto generated RRT demos')

    parser.add_argument('-n', help="The number of demonstrations to run", default=100)
    parser.add_argument('-d', help="Use this flag to indicate the obstacles should be dynamic", action='store_true')
    parser.add_argument('-l', '--label', help="Label for this run", default='')
    args = parser.parse_args()
    main(args)
