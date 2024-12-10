import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from envs.ArmPickAndDrop import ArmPickAndDrop
from envs.robot import Panda, UR5Robotiq85, UR5Robotiq140
from helper.utilities import YCBModels, Camera


def user_control_demo():
    ycb_models = YCBModels(
        os.path.join("./data/ycb", "**", "textured-decmp.obj"),
    )
    camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)
    camera = None
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    target_position_B = np.array([0.5, 0.5, 0.0])
    env = ArmPickAndDrop(
        robot, ycb_models, camera, vis=True, target_position_B=target_position_B
    )

    env.reset()
    # env.SIMULATION_STEP_DELAY = 0
    while True:
        obs, reward, done, info = env.step(env.read_debug_parameter(), "end")

        # print(obs, reward, done, info)


if __name__ == "__main__":
    user_control_demo()
