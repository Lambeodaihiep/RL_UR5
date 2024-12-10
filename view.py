import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from envs.ArmPickAndDrop import ArmPickAndDrop
from envs.robot import UR5Robotiq85
from helper.utilities import YCBModels, Camera

ycb_models = YCBModels(
    os.path.join("./data/ycb", "**", "textured-decmp.obj"),
)
camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)
camera = None
robot = UR5Robotiq85((0.0, 0.0, 0.4), (0, 0, 0))
target_position_B = np.array([0.5, 0.5, 0.0])

env = ArmPickAndDrop(robot, ycb_models, camera, vis=True, target_position_B=target_position_B)
env.reset()

# action = np.array([np.pi/2, 0, 0, 0, 0, 0, 0.005])
action = np.array([-1.5690622952052096,
                    -1.5446774605904932,
                    1.343946009733127,
                    -1.3708613585093699,
                    -1.5707970583733368,
                    0.0009377758247187636,
                    0])

for i in range(1000):
    # env.render()
    # observation, reward, terminated, info = env.step(env.action_space.sample())  # take a random action
    # if info["is_obstacle_collision"]:
    #     env.reset()
    env.step(action)

env.close()
