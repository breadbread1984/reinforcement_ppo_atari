#!/usr/bin/python3

from os import environ
import torch
from torch import nn
from transformers import AutoConfig, Qwen3ForCausalLM

class PolicyNet(nn.Module):
  def __init__(self, action_num, hidden_dim = 8, stack_length = 4):
    super(PolicyNet, self).__init__()
    self.encoding = nn.Sequential(
      nn.Conv2d(stack_length, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h = 8, 112, 112)
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h = 8, 56, 56)
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h = 8, 28, 28)
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h = 8, 14, 14)
      nn.Flatten(), # (b, h * 14 * 14)
      nn.Linear(hidden_dim * 14 * 14, 64) # (b, config.hidden_size)
    )
    self.pred_head = nn.Sequential(
      nn.Linear(64, action_num),
      nn.Softmax(dim = -1)
    )
  def forward(self, x, sample_num = 1):
    encoding = self.encoding(x) # encoding.shape = (b, hidden_size)
    weights = self.pred_head(encoding) # weights.shape = (b, action_num)
    actions = torch.multinomial(weights, sample_num) # action.shape = (b, sample_num)
    logprob = torch.log(torch.gather(weights, dim = -1, index = actions)) # logprob.shape = (b, sample_num)
    return actions, logprob
  def get_probs(self, x, actions):
    encoding = self.encoding(x)
    weights = self.pred_head(encoding)
    logprob = torch.log(torch.gather(weights, dim = -1, index = actions))
    return logprob

class ValueNet(nn.Module):
  def __init__(self, hidden_dim = 8, stack_length = 4):
    super(ValueNet, self).__init__()
    self.valuenet = nn.Sequential(
      nn.Conv2d(stack_length, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Flatten(),
      nn.Linear(hidden_dim * 14 * 14, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, 1)
    )
  def forward(self, x):
    # x.shape = (b, 3, h, w)
    values = self.valuenet(x) # values.shape = (b, 1)
    return values

class PPO(nn.Module):
  def __init__(self, action_num, hidden_dim = 8, stack_length = 4, is_train = False):
    super(PPO, self).__init__()
    self.is_train = is_train
    if self.is_train:
      self.reference_net = PolicyNet(action_num)
      for param in self.reference_net.parameters():
        param.requires_grad = False
    self.policy_net = PolicyNet(action_num, hidden_dim = hidden_dim, stack_length = stack_length)
    self.value_net = ValueNet(hidden_dim, hidden_dim = hidden_dim, stack_length = stack_length)
    self.dist = torch.distributions.Normal
    # synchronize policy net with reference net
    if self.is_train: self.update_ref()
  def act(self, x, sample_num = 1):
    actions, logprob = self.policy_net(x, sample_num = sample_num) # action.shape = (batch, 1), logprob.shape = (batch, 1)
    if self.is_train:
      with torch.no_grad():
        ref_logprob = self.reference_net.get_probs(x, actions, past_key_values = past_key_values)
    # actions.shape = (b, sample_num) logprob.shape = (b, sample_num), ref_logprob.shape = (b, sample_num)
    # NOTE: backpropagation is done through logprob
    return (actions.detach(), logprob, ref_logprob.detach()) if self.is_train else \
           (actions.detach(), logprob)
  def advantages(self, states, rewards, values, dones, gamma = 0.95, lam = 0.95):
    assert states.shape[0] == rewards.shape[0] + 1 == values.shape[0] + 1 == dones.shape[0] + 1
    T = rewards.shape[0]
    advantages = torch.zeros(T)
    advantage = 0
    for t in reversed(range(T)):
      # A_t = r_t - V(s_t) if the trajectory ends at terminal state
      # A_t = r_t + gamma * V(s_{t+1}) - V(s_t) if the trajectory does not ends at terminal state
      delta = rewards[t] + (0 if dones[t] else \
                            gamma * values[t + 1] if t != T - 1 else \
                            gamma * self.value_net(states[-1:])[0,0]) - values[t]
      # GAE_t = A_t + gamma * lambda * GAE_{t+1} if t != T
      # GAE_T = A_t if t == T
      advantages[t] = delta + (0 if dones[t] else \
                               gamma * lam * advantages[t + 1] if t != T - 1 else \
                               0)
    assert advantages.shape[0] == rewards.shape[0]
    # NOTE: do not propagate through advantages
    return advantages.detach()
  def get_values(self, states, rewards, dones, gamma):
    # states.shape = (length + 1, 3, h, w) need s_{t+1} to get V(s_{t+1}) to calculate V(s_t) = r_t + r * V(s_{t+1})
    # rewards.shape = (length)
    # dones.shape = (length)
    assert states.shape[0] == rewards.shape[0] + 1 == dones.shape[0] + 1
    # 1) calculate the V(s_t) of the last state s_t before truncation
    discount_cumsum = torch.zeros_like(rewards).to(next(self.parameters()).device)
    # 2) calculate the leading V(s_t)
    # 2.1) calculate V(s_t) = r_t if the trajectory ends at terminal state
    # V(s_t) = r_t + gamma * V(s_{t+1}) if the trajectory does not end at terminal state
    discount_cumsum[-1] = rewards[-1] + (0 if dones[-1] else gamma * self.value_net(states[-1:])[0,0])
    # 2.2) iterate backward
    for t in reversed(range(rewards.shape[0] - 1)):
      discount_cumsum[t] = rewards[t] + gamma * discount_cumsum[t + 1]
    assert discount_cumsum.shape[0] == rewards.shape[0]
    return discount_cumsum.detach()
  def pred_values(self, states):
    values = self.value_net(states[:-1]) # values.shape = (1, 1)
    values = torch.squeeze(values, dim = -1) # values.shape = (1)
    return values
  def update_ref(self,):
    state = self.policy_net.state_dict()
    self.reference_net.load_state_dict(state)

if __name__ == "__main__":
  policy = PolicyNet(12)
  image = torch.randn(8,3,224, 224)
  action, logprob = policy(image)
  print(action.shape, logprob.shape)
  value = ValueNet(12)
  v = value(image)
  print(v.shape)
