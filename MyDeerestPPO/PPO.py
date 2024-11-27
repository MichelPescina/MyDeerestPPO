from MyDeerestPPO.PPOData import PPOData
from MyDeerestPPO.PPOAgent import PPOAgent
import torch
import torch.nn as nn
import numpy as np

class PPO:
    def __init__(
            self,
            gamma,
            epsilon,
            mb_size,
            optim,
            vloss_const = 0.5,
            clip_gradients = False,
            max_grad_norm = 0.5,
            shuffle = True,
            device = 'cpu'):
        """
        This PPO implementation supports continuous action spaces and recurrent neural networks.
        For recurrent agents you will need to disable shuffling and enable gradient clipping.
        Also you will have to handle the state of your recurrent blocks in the agent class, also
        you will need to reconstruct the recurrent block hidden state before you start the learning
        phase, you have to implement that yourself.
        """
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.optim = optim
        self.mb_size = mb_size
        self.optim = optim
        self.shuffle = shuffle
        self.device = device
        self.clip_gradients = clip_gradients
        self.max_grad_norm = max_grad_norm
        self.vloss_const = vloss_const


    def learn(
            self,
            agent: PPOAgent,
            obs: PPOData,
            rewards: PPOData,
            actions: PPOData,
            logprobs: PPOData,
            resets: list[bool]
        ):
        """
        Expects data with shape [timesteps, n_envs, (shape of your input)]
        """
        
        # Check if all of the data has the same number of timesteps
        assert(len(resets) == obs.shape[0] == rewards.shape[0] == actions.shape[0] == logprobs.shape[0]), "Data doesn't have same number of timesteps"
        batch_size = len(resets)
        assert batch_size % self.mb_size == 0, "Batch size not divisable by mini-batch size."
        # Info variables
        info = {'kl_div': []}
        # Starts here
        V_target = self._discounted_returns(agent, obs, rewards, resets)
        indices = np.arange(batch_size)
        if self.shuffle:
            np.random.shuffle(indices)
        for mb_ind, start in enumerate(range(0, batch_size, self.mb_size)):
            # Creating minibatches
            end = start + self.mb_size
            mb_indices = indices[start:end]
            mb_obs = obs.get_minibatch(mb_indices).to(self.device)
            mb_actions = actions.get_minibatch(mb_indices).to(self.device)
            mb_logprobs = logprobs.get_minibatch(mb_indices).to(self.device)
            mb_curr_logprobs = agent.actor_evaluate(mb_obs, mb_actions)
            # Advantage estimate
            mb_V_target = V_target[mb_indices]
            mb_curr_V = agent.critic_evaluate(mb_obs)
            adv_estim = mb_V_target - mb_curr_V
            # Clipped surrogate loss
            ratio = torch.exp(mb_curr_logprobs - mb_logprobs)
            surr1 = adv_estim * ratio
            surr2 = adv_estim * torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
            clipped_loss = -(torch.min(surr1, surr2).mean())
            # Critic loss
            critic_loss = ((mb_curr_V - mb_V_target) ** 2).mean()
            # Backpropagation and optimizer step
            loss = clipped_loss + self.vloss_const * critic_loss
            self.optim.zero_grad()
            loss.backward()
            if self.clip_gradients:
                nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
            self.optim.step()
            # KL divergence, useful for debugging, if > 0.02 probably a bug exists
            # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
            with torch.no_grad():
                logratio = mb_curr_logprobs - mb_logprobs
                approx_kl = ((ratio - 1) - logratio).mean()
                info["kl_div"].append(approx_kl.item())
        return info


    def _discounted_returns(self, agent:PPOAgent, obs: PPOData, rewards: PPOData, resets: list[bool]):
        with torch.no_grad():
            timesteps = len(resets)
            discounted_returns = torch.zeros(rewards.shape, dtype=rewards.dtype).to(self.device)
            accum = torch.zeros((1, rewards.shape[1],), dtype=rewards.dtype).to(self.device)
            for t in reversed(range(timesteps)):
                if resets[t]:
                    # Bootstrapping
                    V_bootstrap = agent.critic(obs.get(t)).squeeze(dim=1)
                    #print(f"{t}) Bootstraping V: {V_bootstrap.shape} disc: {discounted_returns[t].shape}")
                    discounted_returns[t] = V_bootstrap
                    accum = V_bootstrap * self.gamma
                else:
                    discounted_returns[t] = rewards.data[t] + accum
                    accum = self.gamma * discounted_returns[t]
        return discounted_returns
                    


