import torch
import DeveloperName
import numpy as np

from BehaviorCloning import MLP
from Configuration import DiscreteDirectionAction, StaticObstaclesConfiguration

saved_states = torch.load(DeveloperName.my_name + 'model.pk')
model = MLP(input_shape=4)
model.load_state_dict(saved_states['state_dict'])
model.eval()


class BCRRT:
    """
    This is essentially Max's Algorithm 1 from "Efficient Exploration via First-Person Behavior Cloning Assisted Rapidly-Exploring Random Trees"
    """

    def __init__(self, conf):
        """
        Takes in a BC policy and a start conf
        :param policy:
        """
        self.conf = conf
        self.step = 5
        self.G = [conf]

    def run(self, rollout_episodes: int):
        """
        Rolls out the BC policy for N iters starting from conf and builds a tree, then runs RRT on that tree
        :param rollout_episodes:
        :return:
        """
        while rollout_episodes:
            roll_out_depth = 40
            conf = self.conf
            while roll_out_depth:
                x = torch.tensor(conf.as_vector(), dtype=torch.float, requires_grad=False)
                x = x.view(1, 4)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                predicted = int(predicted)
                predicted += 1  # index start from 0
                action = DiscreteDirectionAction(predicted)
                new_conf = StaticObstaclesConfiguration(conf.agent + (20 * action.direction_vector()),
                                                        conf.goal)

                roll_out_depth -= 1
                if not conf.is_valid_conf(new_conf):
                    continue
                new_conf.parent_vector = (conf, action)
                self.G.append(new_conf)
                conf = new_conf
            rollout_episodes -= 1

        return self.G
