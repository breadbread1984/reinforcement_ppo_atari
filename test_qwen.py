#!/usr/bin/python3

from absl import flags, app
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation
import ale_py
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch import nn
from models_qwen import PPO

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt.pth', help = 'path to trained model')
  flags.DEFINE_enum('game', default = 'box', enum_values = {'box'}, help = 'game to test')
  flags.DEFINE_integer('stack_length', default = 4, help = 'length of the stack')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def preprocess(img):
  # img.shape = (c = 4, H, W)
  data = np.stack([cv2.resize(frame, (224,224)) for frame in img], axis = 0) # img.shape = (4, h, w)
  return data

def main(unused_argv):
  gym.register_envs(ale_py)
  env_id = {
    'box': 'ALE/Boxing-v5'
  }[FLAGS.game]
  env = FrameStackObservation(GrayscaleObservation(gym.make(env_id)), FLAGS.stack_length)
  ppo = PPO(action_num = env.action_space.n).to(FLAGS.device)
  ckpt = torch.load(FLAGS.ckpt, map_location = torch.device(FLAGS.device))
  ppo.load_state_dict(ckpt['state_dict'], weights_only = True)
  obs, info = env.reset()
  done = False
  past_key_values = None
  while not done:
    env.render()
    obs = torch.from_numpy(np.stack([preprocess(obs)], axis = 0).astype(np.float32)).to(next(ppo.parameters()).device)
    actions, logprobs, past_key_values = ppo.act(obs, past_key_values = past_key_values)
    actions, logprobs = actions.cpu(), logprobs.cpu()
    actions = np.squeeze(actions.numpy(), axis = -1) # actions.shape = (1,)
    obs, reward, terminate, truncate, info = env.step(actions[0])
  env.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

