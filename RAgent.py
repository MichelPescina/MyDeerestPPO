import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from Models import BasicBB
from Models.Blocks import layer_init, layer_init_rnn
from MyDeerestPPO.PPORecAgent import PPORecAgent


class RAgent(nn.Module, PPORecAgent):
    def __init__(self, n_actions: int, std_dev = 0.5, device = 'cpu'):
        super().__init__()
        self.device = device
        self.n_actions = n_actions
        self.change_exploration_rate(std_dev)
        self.actor_model = BasicBB()
        self.critic_model = nn.Sequential(
            BasicBB(),
            layer_init(nn.Linear(128, 64)),
            nn.Hardswish(),
            layer_init(nn.Linear(64, 1), 1.0),
        )
        self.rec1 = layer_init_rnn(nn.GRU(128, 128, 1), 1.0)
        self.rec2 = layer_init_rnn(nn.GRU(128, n_actions, 1), 1.0)


    def change_exploration_rate(self, std_dev):
        self.std_dev = std_dev
        cov_var = torch.full(size=(self.n_actions,), fill_value=self.std_dev).to(self.device)
        self.cov_mat = torch.diag(cov_var)

    
    def actor(self, hiddens, obs):
        means = self.actor_model(obs).unsqueeze(dim=0)
        means, h0 = self.rec1(means, hiddens[0])
        means, h1 = self.rec2(means, hiddens[1])
        actions = means
        distribution = MultivariateNormal(means, self.cov_mat)
        if self.training:
            actions = distribution.sample()
        logprob = distribution.log_prob(actions)
        return actions, logprob, [h0, h1]


    def critic(self, obs):
        return self.critic_model(obs)


    def actor_evaluate(self, hiddens, obs, actions, resets):
        n_steps = obs.shape[0]
        n_envs = obs.shape[1]
        h0 = hiddens[0]
        h1 = hiddens[1]
        logprob = torch.zeros((n_steps, n_envs)).to(self.device)
        for t in range(n_steps):
            if resets[t]:
                h0 = torch.zeros_like(h0)
                h1 = torch.zeros_like(h1)
            means = self.actor_model(obs[t]).unsqueeze(dim=0)
            means, h0 = self.rec1(means, h0)
            means, h1 = self.rec2(means, h1)
            distribution = MultivariateNormal(means, self.cov_mat)
            logprob[t] = distribution.log_prob(actions[t])
        return logprob, [h0, h1]


    def critic_evaluate(self, obs):
        n_steps = obs.shape[0]
        n_envs = obs.shape[1]
        obs = torch.reshape(obs, (-1,) + obs.shape[2:])
        values = self.critic_model(obs)
        return values.reshape((n_steps, n_envs))


if __name__ == "__main__":
    test = RAgent(4)
    input = torch.randn(10, 5, 3, 240, 320)
    fake_actions = torch.randn(10, 5, 4)
    fake_resets = [0] * 10
    test.prepare_to_learn([None, None], fake_resets)
    output = test.critic_evaluate(input)
    print(output)