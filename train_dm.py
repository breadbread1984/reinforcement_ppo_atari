#/usr/bin/python3

from absl import flags, app
from os.path import exists, join
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import frame_stack
import ale_py
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from models_dm import PPO

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt.pth', help = 'path to checkpoint')
  flags.DEFINE_enum('game', default = 'box', enum_values = {'box'}, help = 'game to train with')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_string('logdir', default = 'logs', help = 'path to log directory')
  flags.DEFINE_integer('stack_length', default = 4, help = 'length of the stack')
  flags.DEFINE_integer('steps', default = 10000, help = 'number of steps per epoch')
  flags.DEFINE_integer('batch', default = 32, help = 'number of trajectories collected parallely')
  flags.DEFINE_integer('traj_length', default = 256, help = 'maximum length of a trajectory')
  flags.DEFINE_integer('epochs', default = 300, help = 'number of epoch')
  flags.DEFINE_integer('update_ref_n_epochs', default = 4, help = 'update reference model every n epochs')
  flags.DEFINE_float('gamma', default = 0.95, help = 'gamma value')
  flags.DEFINE_float('lam', default = 0.95, help = 'lambda')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def preprocess(img):
  # img.shape = (H, W, c=4)
  img = cv2.resize(img, (224, 224))
  data = np.transpose(img, (2,0,1)) # data.shape = (c, h, w)
  return data

def main(unused_argv):
  gym.register_envs(ale_py)
  env_id = {
    'box': 'ALE/Boxing-v5'
  }[FLAGS.game]
  envs = SyncVectorEnv([lambda: frame_stack(gym.make(env_id), num_stack = FLAGS.stack_length) for _ in range(FLAGS.batch)])
  ppo = PPO(action_num = envs.single_action_space.n, is_train = True).to(FLAGS.device)
  criterion = nn.MSELoss().to(FLAGS.device)
  optimizer = Adam(ppo.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  tb_writer = SummaryWriter(log_dir = FLAGS.logdir)
  global_steps = 0
  if exists(FLAGS.ckpt):
    ckpt = torch.load(FLAGS.ckpt)
    global_steps = ckpt['global_steps']
    ppo.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
  for epoch in tqdm(range(FLAGS.epochs), desc = 'epoch'):
    for step in tqdm(range(FLAGS.steps), desc = 'step', leave = False):
      # 1) collect trajectories
      trajs = [
        {
          'state': list(),
          'logprob': list(),
          'ref_logprob': list(),
          'reward': list(),
          'done': list()
        } for _ in range(FLAGS.batch)
      ]
      obs, info = envs.reset()
      for ob, traj in zip(obs, trajs):
        traj['state'].append(preprocess(ob))
      for step in range(FLAGS.traj_length):
        obs = torch.from_numpy(np.stack([preprocess(ob) for ob in obs], axis = 0).astype(np.float32)).to(next(ppo.parameters()).device) # obs.shape = (n_traj, 3, 224, 224)
        actions, logprobs, ref_logprobs = ppo.act(obs) # actions.shape = (n_traj, 1), logprob.shape = (n_traj, 1) ref_logprob.shape = (n_traj, 1)
        # move to CPU to reduce GPU usage
        actions, logprobs, ref_logprobs = actions.cpu(), logprobs.cpu(), ref_logprobs.cpu()
        actions = np.squeeze(actions.numpy(), axis = -1) # actions.shape = (n_traj)
        obs, rewards, terminates, truncates, infos = envs.step(actions)
        # obs.shape = (n_traj, h, w, 3) rewards.shape = (n_traj) terminates.shape = (n_traj) truncates.shape = (n_traj)
        for ob, reward, logprob, ref_logprob, done, traj in zip(obs, rewards, logprobs, ref_logprobs, terminates, trajs):
          traj['state'].append(preprocess(ob)) # np.ndarray shape = (traj_length + 1, 3, h, w)
          traj['logprob'].append(logprob) # torch.Tensor shape = (traj_length, 1, 1)
          traj['ref_logprob'].append(ref_logprob) # torch.Tensor shape = (traj_length, 1, 1)
          traj['reward'].append(reward) # np.ndarray shape = (traj_length)
          traj['done'].append(done) # np.ndarray shape = (traj_length)
      # 2) train with trajectories
      for traj in trajs:
        states = torch.from_numpy(np.stack(traj['state']).astype(np.float32)).to(next(ppo.parameters()).device) # states.shape = (traj_length + 1, 3, 224, 224)
        logprobs = torch.squeeze(torch.cat(traj['logprob'], dim = 0), dim = -1).to(next(ppo.parameters()).device) # logprobs.shape = (traj_length,)
        ref_logprobs = torch.squeeze(torch.cat(traj['ref_logprob'], dim = 0), dim = -1).to(next(ppo.parameters()).device) # ref_logprobs.shape = (traj_length)
        rewards = torch.from_numpy(np.array(traj['reward']).astype(np.float32)).to(next(ppo.parameters()).device) # rewards.shape = (traj_length)
        dones = torch.from_numpy(np.array(traj['done']).astype(np.float32)).to(next(ppo.parameters()).device) # dones.shape = (traj_length)
        true_values = ppo.get_values(states, rewards, dones, gamma = FLAGS.gamma) # true_values.shape = (traj_length)
        pred_values = ppo.pred_values(states)
        advantages = ppo.advantages(states, rewards, true_values, dones, FLAGS.gamma, FLAGS.lam) # advantages.shape = (traj_length)
        optimizer.zero_grad()
        loss = -torch.mean(logprobs / ref_logprobs * advantages) + 0.5 * criterion(pred_values, true_values)
        loss.backward()
        optimizer.step()
        tb_writer.add_scalar('loss', loss, global_steps)
        global_steps += 1
    scheduler.step()
    if epoch % FLAGS.update_ref_n_epochs == 0:
      ppo.update_ref()
    ckpt = {
      'global_steps': global_steps,
      'state_dict': ppo.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler
    }
    torch.save(ckpt, FLAGS.ckpt)
  envs.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

