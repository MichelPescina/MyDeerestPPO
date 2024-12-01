# System
import pathlib
import time
import datetime
import argparse
import traceback
# Venv Packages
import gymnasium as gym
import FigFollowerEnv
import numpy as np
import torch
# Local
import MyDeerestPPO as Deer
from RAgent import RAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_reward_performance(infos):
    performance = (infos['accum_reward'] / infos['max_reward']).mean()
    return performance


def make_env(gym_id, idx, capture_video, run_name):
    def miau():
        if capture_video and idx == 0:
            env = gym.make(gym_id, render_mode="rgb_array", fps=10, nodes = 6, max_time = 60)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x%10==0)
        else:
            env = gym.make(gym_id, render_mode=None, fps=10, nodes = 6, max_time = 60)
        return env
    return miau


def to_tensor(obs):
    tensor = torch.from_numpy(obs).to(DEVICE)
    tensor = tensor.float() / 255
    return tensor.permute(0, 3, 1, 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=1,
        help="Number of environments of the experiment")
    parser.add_argument("--b_size", type=int, default=100,
        help="Batch size of the data. Must be divisible by mini batch size.")
    parser.add_argument("--mb_size", type=int, default=20,
        help="Mini batch size.")
    parser.add_argument("--steps", type=int, default=1000,
        help="Number of timesteps the experiment will run for")
    parser.add_argument("--video", action='store_true',
        help="Record videos")
    args = parser.parse_args()
    assert args.b_size % args.mb_size == 0, "Batch size not divisable by mini-batch size"
    return args


if __name__ == "__main__":
    args = parse_args()
    # Parameters
    n_envs = args.n_envs
    batch_size = args.b_size
    mb_size = args.mb_size
    total_timesteps = args.steps
    gym_id = 'FigFollowerEnv-v1'
    now = datetime.datetime.now()
    run_name = f'{gym_id}_{now.year}{now.month}{now.day}_{now.hour}{now.minute}'
    # Counters and variables
    steps = 0
    performance = 0
    last_save = 0
    save_period = 2500
    TRAINING = pathlib.Path('Training/' + run_name + '/')
    TRAINING.mkdir(parents=True, exist_ok=True)
    # Creating parts of the algorithm
    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id, i, args.video, run_name) for i in range(n_envs)]
    )
    agent = RAgent(envs.single_action_space.shape[0], 0.1, device=DEVICE).to(DEVICE)
    optim = torch.optim.Adam(agent.parameters(), 0.001)
    ppo = Deer.PPO_Recurrent(0.99, 0.2, mb_size, optim, device = DEVICE, vloss_const=1.0, clip_gradients=True)
    # Creating batches
    batch_observations = Deer.PPOData(n_envs, batch_size, [3, 240, 320], device=DEVICE)
    batch_actions = Deer.PPOData(n_envs, batch_size, list(envs.single_action_space.shape), device=DEVICE)
    batch_logprobs = Deer.PPOData(n_envs, batch_size, device=DEVICE)
    batch_rewards = Deer.PPOData(n_envs, batch_size, device=DEVICE)
    batch_resets = [False] * batch_size
    try:
        next_obs, infos = envs.reset()
        hiddens = [
            torch.zeros((1, n_envs, 128)).to(DEVICE),
            torch.zeros((1, n_envs, 4)).to(DEVICE),
        ]
        print(hiddens[0].shape)
        while steps < total_timesteps:
            # Data collection phase
            with torch.no_grad():
                start_hiddens = [h.clone() for h in hiddens]
                for t in range(batch_size):
                    obs = next_obs
                    input = to_tensor(obs)
                    actions, logprobs, hiddens = agent.actor(hiddens, input)
                    next_obs, rewards, terminations, truncations, infos = envs.step(actions.squeeze().cpu().numpy())
                    ended = np.any(terminations) or np.any(truncations)
                    batch_observations.update(input, t)
                    batch_actions.update(actions, t)
                    batch_logprobs.update(logprobs, t)
                    batch_rewards.update(torch.from_numpy(rewards).to(DEVICE), t)
                    if ended:
                        next_obs, infos = envs.reset()
                        batch_resets[t] = True
                        hiddens = [
                            torch.zeros((1, n_envs, 128)).to(DEVICE),
                            torch.zeros((1, n_envs, 4)).to(DEVICE),
                        ]
            # Learning Phase
            steps += batch_size
            performance = get_reward_performance(infos)
            info = ppo.learn(
                agent,
                start_hiddens,
                batch_observations,
                batch_rewards,
                batch_actions,
                batch_logprobs,
                batch_resets
                )
            print(f"Timesteps: {steps} Average Batch Reward: {batch_rewards.data.mean()}")
            print(f"Approx KL Divergence: {info['kl_div']}")
            print(f"Reward performance: {performance}")
            if steps >= last_save + save_period:
                last_save = steps
                torch.save(agent.state_dict(), str(TRAINING) + f'/Agent_{steps}.pt')
    except Exception as e:
        envs.close()
        print(traceback.format_exc())
    torch.save(agent.state_dict(), str(TRAINING) + f'/Agent_{steps}.pt')
    envs.close()