#!/usr/bin/python3

from absl import flags, app
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation
import ale_py
import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from models_sb3 import CustomCNN, CustomActorCriticPolicy

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('batch', default = 512, help = 'number of trajectories collected parallely')
  flags.DEFINE_string('save_ckpt', default = 'ckpt.zip', help = 'path to output checkpoint')
  flags.DEFINE_string('load_ckpt', default = None, help = 'path to checkpoint resumed')
  flags.DEFINE_enum('game', default = 'box', enum_values = {'box'}, help = 'game to train with')
  flags.DEFINE_integer("steps", default = 1000000, help = 'steps for training')
  flags.DEFINE_integer("save_freq", default = 10000, help = 'save frequency')
  flags.DEFINE_string('save_path', default = 'checkpoints', help = 'checkpoint path')
  flags.DEFINE_integer('stack_length', default = 4, help = 'length of the stack')

def main(unused_argv):
  gym.register_envs(ale_py)
  env_id = {
    'box': 'ALE/Boxing-v5'
  }[FLAGS.game]
  #env = FrameStackObservation(GrayscaleObservation(gym.make(env_id)), FLAGS.stack_length)
  env = VecFrameStack(make_atari_env(env_id, n_envs = FLAGS.batch, seed = 0), n_stack = FLAGS.stack_length)
  if FLAGS.load_ckpt is None:
    model = PPO(
      CustomActorCriticPolicy,
      env,
      policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 128}
      },
      verbose = 1
    )
  else:
    model = PPO.load(FLAGS.load_ckpt)
    model.set_env(env)
  checkpoint_callback = CheckpointCallback(
    save_freq = FLAGS.save_freq,
    save_path = FLAGS.save_path,
    name_prefix = f"ppo_{FLAGS.game}",
    save_replay_buffer = True,
    save_vecnormalize = True,
  )
  model.learn(total_timesteps = FLAGS.steps, callback = checkpoint_callback)
  model.save(FLAGS.save_ckpt)
  #torch.save(model.policy.state_dict(), FLAGS.save_ckpt)

if __name__ == "__main__":
  add_options()
  app.run(main)

