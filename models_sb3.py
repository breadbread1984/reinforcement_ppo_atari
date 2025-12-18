#!/usr/bin/python3

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomCNN(BaseFeaturesExtractor):
  def __init__(self, observation_space, features_dim = 256):
    super(CustomCNN, self).__init__(observation_space, features_dim)
    n_input_channels = observation_space.shape[0]
    self.cnn = nn.Sequential(
      nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Flatten(),
    )
    with torch.no_grad():
      n_flatten = n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
    self.linear = nn.Linear(n_flatten, features_dim)
  def forward(self, observations):
    return self.linear(self.cnn(observations))

class CustomNetwork(nn.Module):
  def __init__(self, feature_dim: int, last_layer_dim_pi: int = 64, last_layer_dim_vf: int = 64):
    super(CustomNetwork, self).__init__()
    self.latent_dim_pi = last_layer_dim_pi
    self.latent_dim_vf = last_layer_dim_vf
    self.policy_net = nn.Sequential(
      nn.Linear(feature_dim, last_layer_dim_pi),
      nn.ReLU()
    )
    self.value_net = nn.Sequential(
      nn.Linear(feature_dim, last_layer_dim_vf),
      nn.ReLU()
    )
  def forward(self, features: torch.Tensor):
    return self.forward_actor(features), self.forward_critic(features)
  def forward_actor(self, features: torch.Tensor):
    return self.policy_net(features)
  def forward_critic(self, features: torch.Tensor):
    return self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
  def _build_mlp_extractor(self) -> None:
    self.mlp_extractor = CustomNetwork(self.features_dim)

