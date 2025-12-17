#!/usr/bin/python3

import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

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
    return self.policy_net(features), self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
  def _build_mlp_extractor(self) -> None:
    self.mlp_extractor = CustomNetwork(self.features_dim)

