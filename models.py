#!/usr/bin/python3

import torch
from torch import nn

class PolicyNet(nn.Module):
  def __init__(self, action_num, hidden_dim = 8):
    super(PolicyNet, self).__init__()
    self.policy = nn.Sequential(
      nn.Conv2d(3, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h, 112, 112)
      nn.ReLU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h, 56, 56)
      nn.ReLU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h, 28, 28)
      nn.ReLU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h, 14, 14)
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(hidden_dim * 14 * 14, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, action_num),
      nn.Softmax(dim = -1)
    )
  def forward(self, x):
    assert x.shape[1] == 3 and x.shape[2] == 224 and x.shape[3] == 224
    weights = self.policy(x) # mean.shape = (batch, action_num)
    action = torch.multinomial(weights, 1) # action.shape = (batch, 1)
    logprob = torch.log(torch.gather(weights, dim = -1, index = action)) # logprob.shape = (batch, 1)
    return action, logprob

class ValueNet(nn.Module):
  def __init__(self, hidden_dim = 8):
    super(ValueNet, self).__init__()
    self.valuenet = nn.Sequential(
      nn.Conv2d(3, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.ReLU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.ReLU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.ReLU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(hidden_dim * 14 * 14, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 1)
    )
  def forward(self, x):
    values = self.valuenet(x)
    return values

class PPO(nn.Module):
  def __init__(self, action_num, hidden_dim = 8):
    super(PPO, self).__init__()
    self.policy_net = PolicyNet(action_num, hidden_dim)
    self.value_net = ValueNet(hidden_dim)
    self.dist = torch.distributions.Normal
  def act(self, x):
    action, logprob = self.policy_net(x) # action.shape = (batch, 1), logprob.shape = (batch, 1)
    return action, logprob
  def advantages(self, states, rewards, values, dones, gamma = 0.95, lam = 0.95):
    assert states.shape[0] == rewards.shape[0] + 1 == values.shape[0] + 1 == dones.shape[0] + 1
    T = len(rewards)
    advantages = torch.zeros(T)
    advantage = 0
    for t in reversed(range(T)):
      delta = rewards[t] + (0 if dones[t] else \
                            gamma * values[t + 1] if t != T - 1 else \
                            gamma * self.value_net(states[-1:])[0,0]) - values[t]
      advantages[t] = delta + (0 if dones[t] else \
                               gamma * lam * advantages[t + 1] if t != T - 1 else \
                               0)
    assert advantages.shape[0] == rewards.shape[0]
    return advantages
  def get_values(self, states, rewards, dones, gamma):
    assert states.shape[0] == rewards.shape[0] + 1 == dones.shape[0] + 1
    # calculate the V(s_t) of the last state s_t before truncation
    discount_cumsum = torch.zeros_like(rewards).to(next(self.parameters()).device) # discount_cumsum.shape = (len)
    discount_cumsum[-1] = rewards[-1] + (0 if dones[-1] else gamma * self.value_net(states[-1:])[0,0])
    # calculate the leading V(s_t)
    for t in reversed(range(rewards.shape[0] - 1)):
      discount_cumsum[t] = rewards[t] + gamma * discount_cumsum[t + 1]
    assert discount_cumsum.shape[0] == rewards.shape[0]
    return discount_cumsum
  def pred_values(self, states):
    values = self.value_net(states[:-1]) # values.shape = (len, 1)
    values = torch.squeeze(values, dim = -1) # values.shape = (len,)
    return values

if __name__ == "__main__":
  policy = PolicyNet(12)
  image = torch.randn(8,3,224, 224)
  action, logprob = policy(image)
  print(action.shape, logprob.shape)
  value = ValueNet(12)
  v = value(image)
  print(v.shape)
