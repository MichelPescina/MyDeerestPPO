import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.distributions.multivariate_normal import MultivariateNormal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_init(layer, std = np.sqrt(2), const_bias = 0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, const_bias)
  return layer


class DWSConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, activation, layer_init = None):
    super(DWSConv2d, self).__init__()
    if type(layer_init) != type(None):
      self.depth = layer_init(nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            groups = in_channels,
            padding = 'same'
          ))
      self.point = layer_init(nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
          ))
    else:
      self.depth = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            groups = in_channels,
            padding = 'same'
          )
      self.point = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
          )
    self.activ = activation

  def forward(self, x):
    x = self.activ(self.depth(x))
    x = self.activ(self.point(x))
    return x


class MiauBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, layer_init = None):
    super().__init__()
    if type(layer_init) != type(None):
      self.block = nn.Sequential(
          DWSConv2d(in_channels, out_channels, kernel_size, nn.Hardswish(), layer_init),
          nn.BatchNorm2d(out_channels),
          layer_init(nn.Conv2d(out_channels, out_channels, kernel_size = 2, stride = 2, groups = out_channels)),
          nn.Hardswish()
      )
    else:
      self.block = nn.Sequential(
          DWSConv2d(in_channels, out_channels, kernel_size, nn.Hardswish(), None),
          nn.BatchNorm2d(out_channels),
          nn.Conv2d(out_channels, out_channels, kernel_size = 2, stride = 2, groups = out_channels),
          nn.Hardswish()
      )

  def forward(self, x):
    return self.block(x)


class Agent(nn.Module):
  def __init__(self, n_actions, in_channels : int, std_dev = 0.5, device = 'cpu', weights = True):
    super().__init__()
    self.device = device
    self.n_actions = n_actions
    self.change_exploration_rate(std_dev)
    # Actor model
    self.actor_model = nn.Sequential(
      MiauBlock(in_channels, in_channels * 2, 5, lambda l: layer_init(l)),
      MiauBlock(in_channels * 2, in_channels * 3, 3, lambda l: layer_init(l)),
      MiauBlock(in_channels * 3, in_channels * 4, 3, lambda l: layer_init(l)),
      MiauBlock(in_channels * 4, in_channels * 5, 3, lambda l: layer_init(l)),
      layer_init(nn.Conv2d(in_channels * 5, n_actions * 5, 1)),
      nn.Tanh(),
      layer_init(nn.Conv2d(n_actions * 5, n_actions * 3, 1)),
      nn.Tanh(),
      layer_init(nn.Conv2d(n_actions * 3, n_actions, 1), 0.01),
      nn.Tanh(),
      nn.AdaptiveAvgPool2d((1,1)),
      nn.Flatten()
    )
    # Critic model
    self.critic_model = nn.Sequential(
      MiauBlock(in_channels, in_channels * 2, 5, lambda l: layer_init(l)),
      MiauBlock(in_channels * 2, in_channels * 3, 3, lambda l: layer_init(l)),
      MiauBlock(in_channels * 3, in_channels * 4, 3, lambda l: layer_init(l)),
      MiauBlock(in_channels * 4, in_channels * 5, 3, lambda l: layer_init(l)),
      layer_init(nn.Conv2d(in_channels * 5, n_actions * 5, 1)),
      nn.Tanh(),
      layer_init(nn.Conv2d(n_actions * 5, n_actions * 3, 1)),
      nn.Tanh(),
      layer_init(nn.Conv2d(n_actions * 3, 1, 1), 1.0),
      nn.Tanh(),
      nn.AdaptiveAvgPool2d((1,1)),
      nn.Flatten()
    )

  def change_exploration_rate(self, std_dev):
    self.std_dev = std_dev
    cov_var = torch.full(size=(self.n_actions,), fill_value=self.std_dev).to(self.device)
    self.cov_mat = torch.diag(cov_var)


  def actor(self, obs):
    means = self.actor_model(obs)
    actions = means
    distribution = MultivariateNormal(means, self.cov_mat)
    if self.actor_model.training:
      actions = distribution.sample()
    log_prob = distribution.log_prob(actions)
    return actions, log_prob

  def actor_evaluate(self, obs, actions):
    means = self.actor_model(obs)
    distribution = MultivariateNormal(means, self.cov_mat)
    log_prob = distribution.log_prob(actions)
    return log_prob

  def critic(self, obs):
    return self.critic_model(obs)



class Trajectory:
  def __init__(self):
    self.states = []
    self.rewards = []
    self.actions = []
    self.log_probs = []
    self.length = 0
    self.finished = False
    self.next_obs = None

  def record(self, state, reward, action, log_prob):
    self.states.append(state)
    self.rewards.append(reward)
    self.actions.append(action)
    self.log_probs.append(log_prob)
    self.length += 1

  def get_states(self) -> torch.tensor:
    tensor = torch.tensor(np.array(self.states))
    #print(f"states = {tensor.shape}")
    return tensor

  def get_actions(self) -> torch.tensor:
    tensor = torch.tensor(np.array(self.actions))
    #print(f"actions = {tensor.shape}")
    return tensor

  def get_log_probs(self) -> torch.tensor:
    tensor = torch.tensor(np.array(self.log_probs))
    #print(f"logprobs = {tensor.shape}")
    return tensor

  def get_rewards(self) -> list:
    return self.rewards


# This implementation works for tasks that are continuous and agents with separate models for the actor and critic network
class PPO_Continuous:
  def __init__(self, discount_factor = 0.99, epsilon = 0.2):
    # Default Hyperparameters
    self.discount_factor = discount_factor
    self.epsilon = epsilon

  def learn(self, agent, actor_optim, critic_optim, trajectory : Trajectory, minibatch_size):
    # Preparing stuff
    critic_loss_fn = nn.MSELoss()

    batch_states = trajectory.get_states().to(DEVICE)
    batch_log_probs = trajectory.get_log_probs().to(DEVICE)
    batch_actions = trajectory.get_actions().to(DEVICE)
    batch_rewards = trajectory.get_rewards()

    # If the trajectory is from an episode that was truncated then the algorithm computes
    # the estimated reward for the remainder of the episode.
    with torch.no_grad():
      bootstrap_V = 0.0
      if not trajectory.finished:
        bootstrap_V = agent.critic(batch_states[-1].unsqueeze(dim=0)).to('cpu').item()
      V_target = self.discounted_rewards(trajectory.rewards, self.discount_factor, bootstrap_V).to(DEVICE)

    # Shuffle the indexes
    indices = np.arange(trajectory.length)
    np.random.shuffle(indices)

    # Start training on minibatches
    for mb_ind, start in enumerate(range(0, trajectory.length, minibatch_size)):
      # Initializes stuff for the batch
      end = start + minibatch_size
      batch_indices = indices[start:end]
      if len(batch_indices) < 2:
        continue
      mb_states = batch_states[batch_indices]
      mb_actions = batch_actions[batch_indices]
      mb_prev_log_probs = batch_log_probs[batch_indices]
      mb_V_target = V_target[batch_indices]
      # Gets the value estimates given the current policy
      mb_V_current = torch.squeeze(agent.critic(mb_states))
      # Gets the logarithmic probabilities of actions taken at state t according to
      # the current policy
      mb_curr_log_probs = agent.actor_evaluate(mb_states, mb_actions)
      # Advantage estimate
      adv_estim = mb_V_target - mb_V_current
      #print(f"mb_states: {mb_states.shape}")
      #print(f"mb_actions: {mb_actions.shape}")
      #print(f"mb_prev_log_probs: {mb_prev_log_probs.shape}")
      #print(f"mb_V_target: {mb_V_target.shape}")
      #print(f"mb_V_current: {mb_V_current.shape}")
      #print(f"mb_curr_log_probs: {mb_curr_log_probs.shape}")
      # Policy ratio
      ratio = torch.exp(mb_curr_log_probs - mb_prev_log_probs)
      with torch.no_grad():
        logratio = mb_curr_log_probs - mb_prev_log_probs
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        print("Approx KL Divergence: ", approx_kl.item())
      # Clipped surrogate loss
      surr1 = adv_estim * ratio
      surr2 = adv_estim * torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
      clipped_surrogate_loss = -(torch.min(surr1, surr2).mean())
      #print(f"LClip: {clipped_surrogate_loss} ratio: {ratio.shape} surr1: {surr1.shape} surr2: {surr2.shape}")
      # Backpropagation and optimize
      actor_optim.zero_grad()
      clipped_surrogate_loss.backward(retain_graph=True)
      actor_optim.step()

      critic_loss = critic_loss_fn(mb_V_current, mb_V_target)
      critic_optim.zero_grad()
      critic_loss.backward()
      critic_optim.step()
      #print(f"MB index: {mb_ind} Start: {start} End: {end} Ratio: :3 LClip: {clipped_surrogate_loss} CriticLoss: {critic_loss}")

  def discounted_rewards(self, rewards, discount_factor, bootstrap_V = 0.0) -> torch.tensor:
    length = len(rewards)
    discounted_returns = torch.zeros(length)
    accum = discount_factor * bootstrap_V
    for t in reversed(range(length)):
      discounted_returns[t] = rewards[t] + accum
      accum = discount_factor * (rewards[t] + accum)
    return discounted_returns


class ImageStack:
  def __init__(self, size):
    self.images = []
    self.size = size

  def update(self, img : np.array):
    if len(self.images) == 0:
      for i in range(self.size):
        self.images.append(img)
    temp = self.images.pop()
    del(temp)
    self.images.insert(0, img)

  def get(self):
    return np.stack(self.images)

if __name__ == "__main__":
  import gymnasium as gym
  import FigFollowerEnv
  import numpy as np
  import traceback
  
  env = gym.make('FigFollowerEnv/FigFollower-v1', render_mode="rgb_array", width=320, height=240, fps=7, nodes = 5)
  
  # Wrapping yummy code
  """
  env = gym.wrappers.RecordVideo(
      env=env,
      video_folder = "Videos",
      name_prefix="test_video",
      episode_trigger=lambda x: x % 3 == 0
      )
  """
  env = gym.wrappers.GrayscaleObservation(env)
  env = gym.wrappers.NormalizeReward(env)
  
  # Creating Agent and PPO learning scheme
  stack_size = 12
  agent = Agent(env.action_space.shape[0], stack_size, std_dev = 0.2, device = DEVICE).to(DEVICE)
  ppo = PPO_Continuous(discount_factor=0.95, epsilon=0.2)
  actor_optim = optim.Adam(agent.actor_model.parameters(), lr = 0.0001)
  critic_optim = optim.SGD(agent.critic_model.parameters(), lr = 0.001)
  
  continue_learning = False
  if continue_learning:
    agent.load_state_dict(torch.load('/content/agent_model_80000.pt', weights_only = True))
  
  # Initialization
  episode = 0
  timesteps = 0
  total_timesteps = 500
  learning_interval = 100
  ts_since_last_learning = 0
  minibatch_size = 25
  stack = ImageStack(stack_size)
  
  
  # Reset for fresh start
  obs, info = env.reset()
  next_obs = obs
  trajectory = Trajectory()
  
  try:
    # Start training
    while timesteps <= total_timesteps:
      # Interacting with the environment
      obs = next_obs
      stack.update(obs)
      input = stack.get()
      input = input.astype(np.float32)/255
      tensor_in = torch.from_numpy(input).to(DEVICE).unsqueeze(dim=0)
      with torch.no_grad():
        action, log_prob = agent.actor(tensor_in)
      action = action.squeeze()
      action = action.to('cpu').numpy().astype(float)
      log_prob = log_prob.to('cpu')
      next_obs, reward, terminated, truncated, info = env.step(action)
      trajectory.record(input, reward, action, log_prob)
      if truncated or terminated:
        next_obs, info = env.reset()
        trajectory.finished = True
        episode += 1
        print(f"Episode {episode - 1} Completed! Total timesteps so far: {timesteps} Reward obtained: {np.array(trajectory.rewards).sum()}")
      ts_since_last_learning += 1
      timesteps += 1
      # Learning on the trajectories
      if ts_since_last_learning >= learning_interval or trajectory.finished:
        print(f"Episode {episode}) Total timesteps so far: {timesteps} Reward obtained: {np.array(trajectory.rewards).sum()}")
        ppo.learn(agent, actor_optim, critic_optim, trajectory, minibatch_size)
        ts_since_last_learning = 0
        del(trajectory)
        trajectory = Trajectory()
    if trajectory.length > 10:
      ppo.learn(agent, actor_optim, critic_optim, trajectory, minibatch_size)
  except Exception as e:
    print(e)
    print(traceback.format_exc())
  # The end
  env.close()