import datetime
import argparse
import gymnasium as gym
import numpy as np
import MyDeerestPPO as Deer
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

import traceback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module, Deer.PPOAgent):
    def __init__(self, inputs: int, outputs: int, std_dev = 0.5, device = 'cpu'):
        super().__init__()
        self.device = device
        self.n_actions = outputs
        self.change_exploration_rate(std_dev)
        self.actor_model = nn.Sequential(
            layer_init(nn.Linear(inputs, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16, outputs), 0.01),
            nn.Tanh()
        )
        self.critic_model = nn.Sequential(
            layer_init(nn.Linear(inputs, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16, 1), 1.0),
            nn.Tanh()
        )
    

    def change_exploration_rate(self, std_dev):
        self.std_dev = std_dev
        cov_var = torch.full(size=(self.n_actions,), fill_value=self.std_dev).to(self.device)
        self.cov_mat = torch.diag(cov_var)

    
    def actor(self, obs):
        means = self.actor_model(obs)
        actions = means
        distribution = MultivariateNormal(means, self.cov_mat)
        if self.training:
            actions = distribution.sample()
        logprob = distribution.log_prob(actions)
        return actions, logprob
    

    def critic(self, obs):
        return self.critic_model(obs)

    
    def actor_evaluate(self, obs, actions):
        n_steps = obs.shape[0]
        n_envs = obs.shape[1]
        obs.reshape((-1,) + obs.shape[2:])
        means = self.actor_model(obs)
        distribution = MultivariateNormal(means, self.cov_mat)
        logprob = distribution.log_prob(actions)
        return logprob.reshape((n_steps, n_envs,))
    

    def critic_evaluate(self, obs):
        n_steps = obs.shape[0]
        n_envs = obs.shape[1]
        obs.reshape((-1,) + obs.shape[2:])
        values = self.critic_model(obs)
        return values.reshape((n_steps, n_envs))
    

def make_env(gym_id, idx, capture_video, run_name):
    def miau():
        if capture_video and idx == 0:
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x%50==0)
        else:
            env = gym.make(gym_id, render_mode=None)
        return env
    return miau

        

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
    gym_id = 'Pendulum-v1'
    now = datetime.datetime.now()
    run_name = f'{gym_id}_{now.year}{now.month}{now.day}_{now.hour}{now.minute}'
    # Counters
    steps = 0
    # Creating parts of the algorithm
    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id, i, args.video, run_name) for i in range(n_envs)]
    )
    agent = Agent(envs.single_observation_space.shape[0], envs.single_action_space.shape[0], 0.1, device=DEVICE).to(DEVICE)
    optim = torch.optim.Adam(agent.parameters(), 0.001)
    ppo = Deer.PPO(0.99, 0.2, mb_size, optim, device = DEVICE, vloss_const=1.0)
    # Creating batches
    batch_observations = Deer.PPOData(n_envs, batch_size, list(envs.single_observation_space.shape), device=DEVICE)
    batch_actions = Deer.PPOData(n_envs, batch_size, list(envs.single_action_space.shape), device=DEVICE)
    batch_logprobs = Deer.PPOData(n_envs, batch_size, device=DEVICE)
    batch_rewards = Deer.PPOData(n_envs, batch_size, device=DEVICE)
    batch_resets = [False] * batch_size
    try:
        next_obs, infos = envs.reset()
        while steps < total_timesteps:
            # Data collection phase
            with torch.no_grad():
                for t in range(batch_size):
                    obs = next_obs
                    actions, logprobs = agent.actor(torch.from_numpy(obs).to(DEVICE))
                    next_obs, rewards, terminations, truncations, infos = envs.step((actions*2).cpu().numpy())
                    ended = np.any(terminations) or np.any(truncations)
                    batch_observations.update(torch.from_numpy(obs).to(DEVICE), t)
                    batch_actions.update(actions, t)
                    batch_logprobs.update(logprobs, t)
                    batch_rewards.update(torch.from_numpy(rewards).to(DEVICE), t)
                    if ended:
                        next_obs, infos = envs.reset()
                        batch_resets[t] = True
            # Learning Phase
            steps += batch_size
            info = ppo.learn(
                agent,
                batch_observations,
                batch_rewards,
                batch_actions,
                batch_logprobs,
                batch_resets
                )
            print(f"Timesteps: {steps} Average Batch Reward: {batch_rewards.data.mean()}")
            print(f"Approx KL Divergence: {info['kl_div']}")
    except Exception as e:
        envs.close()
        print(traceback.format_exc())
    envs.close()