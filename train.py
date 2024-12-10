import os
import torch
import numpy as np
import pybullet as p

from envs.ArmPickAndDrop import ArmPickAndDrop
from envs.robot import UR5Robotiq85
from helper.utilities import YCBModels, Camera
from rl_algorithm.TD3 import TD3, ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "./saved_models/"
# os.makedirs(MODEL_SAVE_PATH, exist_ok=False)

def td3_trainning():
    camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)
    robot = UR5Robotiq85((0.0, 0.0, 0.4), (0, 0, 0))
    target_position_B = np.array([0.0, 0.5, 0.0])
    ycb_models = YCBModels(
        os.path.join("./data/ycb", "**", "textured-decmp.obj"),
    )
    env = ArmPickAndDrop(
        robot, ycb_models, camera, vis=False, target_position_B=target_position_B
    )

    state = env.reset()
    state_dim = len(state)
    action_dim = 7  # robot.get_action_space()
    max_action = 1.0
    td3 = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    num_episodes = 1000
    replay_buffer = ReplayBuffer()
    batch_size = 256
    max_steps = 200

    model_filename = os.path.join(MODEL_SAVE_PATH, "td3_model.pth")
    start_episode = 0
    if os.path.exists(model_filename):
        checkpoint = torch.load(model_filename)
        td3.actor.load_state_dict(checkpoint["actor"])
        td3.critic.load_state_dict(checkpoint["critic"])
        td3.actor_target.load_state_dict(checkpoint["actor_target"])
        td3.critic_target.load_state_dict(checkpoint["critic_target"])
        replay_buffer = checkpoint["replay_buffer"]

        start_episode = checkpoint["episode"] + 1
        print(f"Resuming training from episode {start_episode}.")
    # action_now = 0
    for episode in range(num_episodes+1):
        state = env.reset()
        steps = 0
        done = False
        joint_angle = np.zeros(7)
        episode_reward = 0
        while not done and steps < max_steps:
            action = td3.select_action(
                torch.tensor(state, dtype=torch.float32, device=device)
            )

            next_state, reward, done, info = env.step(action)
            # print(reward)
            replay_buffer.add(
                torch.tensor(state, dtype=torch.float32, device=device),
                torch.tensor(action, dtype=torch.float32, device=device),
                torch.tensor(next_state, dtype=torch.float32, device=device),
                torch.tensor(reward, dtype=torch.float32, device=device),
                torch.tensor(done, dtype=torch.float32, device=device),
            )

            if len(replay_buffer) > batch_size:
                td3.train(replay_buffer, batch_size)
                # kiểm tra training bằng gpu hay cpu
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state = next_state
            episode_reward += reward
            steps += 1

        print(f"Episode {episode}, Reward: {episode_reward}")
        if episode % 50 == 0 and episode != 0:
            checkpoint = {
                "episode": start_episode + episode,
                "actor": td3.actor.state_dict(),
                "critic": td3.critic.state_dict(),
                "actor_target": td3.actor_target.state_dict(),
                "critic_target": td3.critic_target.state_dict(),
                "replay_buffer": replay_buffer,
            }
            # # Đặt tên file cho mô hình theo episode
            # model_filename = f"{MODEL_SAVE_PATH}/td3_model_episode_{episode}.pth"

            # Lưu checkpoint
            torch.save(checkpoint, model_filename)
            print(f"Model saved at episode {episode}")


if __name__ == "__main__":
    td3_trainning()
