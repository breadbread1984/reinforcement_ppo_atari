#/usr/bin/python3

from absl import flags, app
from os.path import exists, join
import gymnasium as gym
import ale_py
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from models import PPO

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt.pth', help = 'path to checkpoint')
  flags.DEFINE_enum('game', default = 'box', enum_values = {'box'}, help = 'game to train with')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_string('logdir', default = 'logs', help = 'path to log directory')
  flags.DEFINE_integer('epochs', default = 300, help = 'number of epoch')
  flags.DEFINE_integer('update_ref_n_epochs', default = 4, help = 'update reference model every n epochs')
  flags.DEFINE_integer('episodes', default = 10000, help = 'episodes per epoch')
  flags.DEFINE_integer('max_ep_steps', default = 300, help = 'max episode steps')
  flags.DEFINE_boolean('visualize', default = False, help = 'whether to visualize')
  flags.DEFINE_float('gamma', default = 0.95, help = 'gamma value')
  flags.DEFINE_float('lam', default = 0.95, help = 'lambda')

def preprocess(img):
  img = cv2.resize(img, (224, 224))
  data = np.transpose(img, (2,0,1)) # data.shape = (c, h, w)
  return data

def main(unused_argv):
  gym.register_envs(ale_py)
  env_id = {
    'box': 'ALE/Boxing-v5'
  }[FLAGS.game]
  env = gym.make(env_id, render_mode = "rgb_array")
  reference = PPO(action_num = env.action_space.n)
  ppo = PPO(action_num = env.action_space.n)
  reference.load_state_dict(ppo.state_dict())
  criterion = nn.MSELoss()
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
  for epoch in tqdm(range(FLAGS.epochs)):
    for episode in tqdm(range(FLAGS.episodes), leave = False):
      states, logprobs, ref_logprobs, rewards, dones = list(), list(), list(), list(), list()
      obs, info = env.reset()
      states.append(preprocess(obs)) # s_t
      for step in range(FLAGS.max_ep_steps):
        obs = torch.from_numpy(np.expand_dims(preprocess(obs), axis = 0).astype(np.float32)).to(next(ppo.parameters()).device) # obs.shape = (1, 3, 224, 224)
        _, logprob = ppo.act(obs) # action.shape = (1,4), logprob.shape = (1,1)
        action, ref_logprob = reference.act(obs) # ref_logprob.shape = (1,1)
        obs, reward, done, truc, info = env.step(action.detach().numpy().item()) # r_t
        logprobs.append(logprob)
        ref_logprobs.append(ref_logprob)
        rewards.append(reward)
        dones.append(done)
        if FLAGS.visualize:
          image = env.reader()[:,:,::-1]
          cv2.imshow("", image)
          cv2.waitKey(1)
          print(reward)
        if done or step == FLAGS.max_ep_steps - 1:
          # episodes are truncated to at most 300 steps
          assert len(states) == len(rewards) == len(dones)
          states.append(preprocess(obs)) # save s_t+1
          states = torch.from_numpy(np.stack(states).astype(np.float32)).to(next(ppo.parameters()).device) # states.shape = (len + 1, 3, 224, 224)
          logprobs = torch.cat(logprobs, dim = 0).to(next(ppo.parameters()).device) # logprobs.shape = (len)
          ref_logprobs = torch.cat(ref_logprobs, dim = 0).to(next(ppo.parameters()).device) # ref_logprobs.shape = (len)
          rewards = torch.from_numpy(np.array(rewards).astype(np.float32)).to(next(ppo.parameters()).device) # rewards.shape = (len)
          dones = torch.from_numpy(np.array(dones).astype(np.float32)).to(next(ppo.parameters()).device) # rewards.shape = (len)
          true_values = ppo.get_values(states, rewards, dones, gamma = FLAGS.gamma) # true_values.shape = (len)
          pred_values = ppo.pred_values(states) # pred_values.shape = (len)
          advantages = ppo.advantages(states, rewards, true_values, dones, FLAGS.gamma, FLAGS.lam) # advantages.shape = (len)
          break
        states.append(preprocess(obs))
      # update policy and value networks
      optimizer.zero_grad()
      loss = -torch.mean(logprobs / ref_logprobs * advantages) + 0.5 * criterion(pred_values, true_values)
      loss.backward()
      optimizer.step()
      tb_writer.add_scalar('loss', loss, global_steps)
      global_steps += 1
    scheduler.step()
    if epoch % FLAGS.update_ref_n_epochs == 0:
      reference.load_state_dict(ppo.state_dict())
    ckpt = {
      'global_steps': global_steps,
      'state_dict': ppo.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler
    }
    torch.save(ckpt, FLAGS.ckpt)
  env.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

