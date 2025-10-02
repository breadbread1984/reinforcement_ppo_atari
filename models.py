#!/usr/bin/python3

from os import environ
import torch
from torch import nn
from transformers import AutoConfig, Qwen3ForCausalLM

class PolicyNet(nn.Module):
  def __init__(self, action_num):
    super(PolicyNet, self).__init__()
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    config = AutoConfig.from_pretrained('Qwen/Qwen3-0.6B')
    self.encoding = nn.Sequential(
      nn.Conv2d(3, 8, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h = 8, 112, 112)
      nn.GELU(),
      nn.Conv2d(8, 8, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h = 8, 56, 56)
      nn.GELU(),
      nn.Conv2d(8, 8, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h = 8, 28, 28)
      nn.GELU(),
      nn.Conv2d(8, 8, kernel_size = (3,3), stride = (2,2), padding = 1), # (b, h = 8, 14, 14)
      nn.Flatten(), # (b, h * 14 * 14)
      nn.Linear(8, config.hidden_size) # (b, config.hidden_size)
    )
    self.model = Qwen3ForCausalLM(config)
    self.pred_head = nn.Sequential(
      nn.Linear(config.hidden_size, action_num),
      nn.Softmax(dim = -1)
    )
  def forward(self, x, past_key_values = None, sample_num = 1):
    encoding = self.encoding(x) # encoding.shape = (b, hidden_size)
    encoding = encoding[:,None,:] # encoding.shape = (b, 1, hidden_size)
    results = self.model(inputs_embeds = encoding, past_key_values = past_key_values, use_cache = True)
    hidden = results.last_hidden_state # last_hidden_state.shape = (b, 1, hidden_size)
    new_past_key_values = results.past_key_values
    hidden = torch.squeeze(hidden, dim = 1) # hidden.shape = (b, hidden_size)
    weights = self.pred_head(hidden) # weights.shape = (b, action_num)
    actions = torch.multinomial(weights, sample_num) # action.shape = (b, sample_num)
    logprob = torch.log(torch.gather(weights, dim = -1, index = actions)) # logprob.shape = (b, sample_num)
    return action, logprob, new_past_key_values
  def get_probs(self, x, actions, past_key_values = None):
    encoding = self.encoding(x)
    encoding = encoding[:,None,:]
    results = self.model(inputs_embeds = encoding, past_key_values = past_key_values, use_cache = True)
    hidden = results.last_hidden_state
    hidden = torch.squeeze(hidden, dim = 1)
    weights = self.pred_head(hidden)
    logprob = torch.log(torch.gather(weights, dim = -1, index = action))
    return logprob

class GRPO(nn.Module):
  def __init__(self, action_num, hidden_dim = 8):
    super(PPO, self).__init__()
    self.reference_net = PolicyNet(action_num, hidden_dim)
    self.policy_net = PolicyNet(action_num, hidden_dim)
    self.dist = torch.distributions.Normal
    # synchronize policy net with reference net
    self.update_ref()
  def act(self, x, past_key_values = None, sample_num = 1):
    actions, logprob, new_past_key_values = self.policy_net(x, past_key_values = past_key_values, sample_num = sample_num) # action.shape = (batch, 1), logprob.shape = (batch, 1)
    ref_logprob = self.reference_net.get_probs(x, actions, past_key_values = past_key_values)
    return actions, logprob, ref_logprob, new_past_key_values
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
