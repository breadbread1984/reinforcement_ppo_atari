#!/usr/bin/python3

from absl import flags, app
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation
import ale_py
import torch
from stable_baselnes3 import PPO
from models_sb3 import CustomActorCriticPolicy

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt.pth', help = 'path to checkpoint')
  flags.DEFINE_enum('game', default = 'box', enum_values = {'box'}, help = 'game to train with')  flags.DEFINE_integer("steps", default = 5000, help = 'steps for training')

def main(unused_argv):
  gym.register_envs(ale_py)
  env_id = {
    'box': 'ALE/Boxing-v5'
  }[FLAGS.game]
  env = FrameStackObservation(GrayscaleObservation(gym.make(env_id)), FLAGS.stack_length)
  model = PPO(CustomActorCriticPolicy, env, verbose = 1)
  model.learn(FLAGS.steps)
  torch.save(model.state_dict(), FLAGS.ckpt)

if __name__ == "__main__":
  add_options()
  app.run(main)

