#!/usr/bin/python3

from absl import flags, app
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation
import ale_py
import numpy as np
import cv2
from stable_baselines3 import PPO

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt.zip', help = 'path to trained model')
  flags.DEFINE_enum('game', default = 'box', enum_values = {'box'}, help = 'game to test')
  flags.DEFINE_integer('stack_length', default = 4, help = 'length of the stack')

def main(unused_argv):
  gym.register_envs(ale_py)
  env_id = {
    'box': 'ALE/Boxing-v5'
  }[FLAGS.game]
  env = FrameStackObservation(GrayscaleObservation(gym.make(env_id, render_mode='rgb_array')), FLAGS.stack_length)
  model = PPO.load(FLAGS.ckpt, env = env)
  obs, info = env.reset()
  done = None
  while not done:
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    cv2.imshow('display', frame)
    cv2.waitKey(40)
  env.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

