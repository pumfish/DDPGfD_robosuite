import os
import os.path as osp
import sys
import time
import numpy as np
import joblib
import torch
import torch.nn as nn
import logging
from logger import logger_setup
sys.path.append('/workspace/S/heguanhua2/robot_rl/robosuite_jimu')

## env
import robosuite as suite
import robosuite.macros as macros
from robosuite.wrappers import Wrapper
from robosuite.controllers import load_controller_config
macros.IMAGE_CONVENTION = "opencv"

## agent
from agent import DDPGfDAgent, DATA_RUNTIME, DATA_DEMO
from training_utils import TrainingProgress, timeSince, load_conf, check_path

# use loggers
DEBUG_LLV = 5
loggers = ['RLTrainer', 'DDPGfD', 'TP']
logging_level = logging.DEBUG


## Rewrite the Jimu reset() & step()
class JimuWrapper(Wrapper):
    def __init__(self, env):
        assert env.__class__.__name__ == 'Jimu', \
                "Only support Jimu environment"
        self._env = env

    def _fetch_obs(self, obs):
        # original observation info
        eef_pos = obs['robot0_eef_pos']               # (3, )
        eef_quat = obs['robot0_eef_quat']             # (4, )
        gripper_qpos = obs['robot0_gripper_qpos']     # (6, )

        # get block position
        achieved_goal, desired_goal = self._env.get_cube_pos()

        return np.r_[eef_pos, eef_quat, gripper_qpos,
                     achieved_goal, desired_goal]     # (19, )

    def reset(self):
        obs = self._env.reset()
        obs = self._fetch_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._fetch_obs(obs)
        return obs, reward, done, info


## process observation
# def fetch_obs(env, obs):
#     # original observation info
#     joint_pos_cos = obs['robot0_joint_pos_cos']      #(6,)
#     joint_pos_sin = obs['robot0_joint_pos_sin']      #(6,)
#     eef_pos = obs['robot0_eef_pos']                  #(3,)
#     eef_quat = obs['robot0_eef_quat']                #(4,)
#     gripper_qpos = obs['robot0_gripper_qpos']        #(6,)
#
#     # get jimu env cube position info
#     assert env.__class__.__name__ == 'Jimu', "Only support Jimu environment"
#     achieved_goal, desired_goal = env.get_cube_pos()
#
#     return np.r_[eef_pos, eef_quat, gripper_qpos,
#                  achieved_goal, desired_goal]    #(19,)


## generate Ornstein-Uhlenbeck Noise, add action diversity
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt +\
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prey = x
        return x

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".\
                format(self.mu, self.sigma)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None \
                      else np.zeros_like(self.mu)


## RL Trainer
class RLTrainer:
    def __init__(self, conf_path, eval=False):
        # load conf
        self.full_conf = load_conf(conf_path)
        self.conf = self.full_conf.train_config
        self.env_conf = self.full_conf.env_config

        # init dir and logger
        current_dir = osp.dirname(osp.abspath(__file__))
        progress_dir = osp.join(current_dir, 'progress')
        result_dir = osp.join(current_dir, 'result')

        self.tp = TrainingProgress(progress_dir, result_dir, self.conf.exp_name)
        log_file = osp.join(self.tp.result_path, self.conf.exp_name + '-log.txt')
        logger_setup(log_file, loggers, logging_level)
        self.logger = logging.getLogger('RLTrainer')

        # set device
        self.device = self.conf.device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        self.logger.info("Use CUDA Device" + self.device)

        # set seed
        if self.conf.seed == -1:
            self.conf.seed = os.getpid() + \
                             int.from_bytes(os.urandom(4), byteorder="little") >> 1
        self.logger.info(f"Random Seed = {self.conf.seed}")

        torch.manual_seed(self.conf.seed) # cpu
        np.random.seed(self.conf.seed)

        # backup training config
        if not eval:
            self.tp.backup_file(conf_path, 'traing.yaml')

        # init env
        controller_config = load_controller_config(default_controller="OSC_POSE")
        env = suite.make(env_name='Jimu',
                         robots='UR5e',
                         controller_configs=controller_config,
                         control_freq=self.env_conf.control_freq, #20,
                         horizon=self.env_conf.horizon, #100,
                         use_object_obs=True,
                         use_camera_obs=True,
                         camera_names='frontview',
                         camera_depths=True,
                         camera_heights=256,
                         camera_widths=256,
                         reward_shaping=self.env_conf.control_freq,)
        action_dim = env.action_dim
        self.env = JimuWrapper(env)
        self.logger.info("Environment Loaded")


        # init agent
        agent_config = self.full_conf.agent_config
        agent_config['state_dim'] = 19
        agent_config['action_dim'] = action_dim
        self.agent = DDPGfDAgent(agent_config, self.device)
        self.agent.to(self.device)

        # resume
        if self.conf.restore:
            self.restore_progress(eval)
        else:
            self.episode = 1

        # optimzer & loss func
        self.set_optimizer()

        reduction = 'none'
        if self.conf.mse_loss:
            self.q_criterion = nn.MSELoss(reduction=reduction)
        else:
            self.q_criterion = nn.SmoothL1Loss(reduction=reduction)

        self.demo2memory()
        #TODO: action dim
        self.action_noise = OrnsteinUhlenbeckActionNoise(
                            np.zeros(self.full_conf.agent_config.action_dim),
                            self.full_conf.agent_config.action_noise_std)

    ## resume function
    def restore_progress(self, eval=False):
        # tps for restore process from conf
        self.tp.restore_progress(self.conf.tps)
        # load state dict
        self.agent.actor_b.load_state_dict(
                self.tp.restore_model_weight(self.conf.tps, self.device, prefix='actor_b'))
        self.agent.actor_t.load_state_dict(
                self.tp.restore_model_weight(self.conf.tps, self.device, prefix='actor_t'))
        self.agent.critic_b.load_state_dict(
                self.tp.restore_model_weight(self.conf.tps, self.device, prefix='critic_b'))
        self.agent,critic_t.load_state_dict(
                self.tp.restore_model_weight(self.conf.tps, self.device, prefix='critic_t'))

        # resume from last episode
        self.episode = self.tp.get_meta('saved_episode') + 1
        np.random.set_state(self.tp.get_meta('np.random_state'))
        torch.random.set_rng_state(self.tp.get_meta('torch_random_state'))
        self.logger.info(f"Restore Progress from Progress {self.episode - 1}")


    ## set optimizer
    def set_optimizer(self):
        # set actor & critic optimizer
        self.optimizer_actor = torch.optim.Adam(
                self.agent.actor_b.parameters(),
                lr=self.conf.lr_rate,
                weight_decay=self.conf.w_decay)
        self.optimizer_critic = torch.optim.Adam(
                self.agent.critic_b.parameters(),
                lr=self.conf.lr_rate,
                weight_decay=self.conf.w_decay)


    ## read the demo data to memory buffer
    def demo2memory(self):
        demo_conf = self.full_conf.demo_config
        if demo_conf.load_demo_data:
            for f_idx in range(demo_conf.load_N):
                self.agent.episode_reset()
                #TODO: demo pkl format
                fname = osp.join(
                        demo_conf.demo_dir,
                        demo_conf.prefix + str(f_idx) + '.pkl')
                data = joblib.load(fname)
                for exp in data:
                    #TODO demo data align
                    s, a, r, s2, done = exp
                    s_tensor = torch.from_numpy(s).float()
                    s2_tensor = torch.from_numpy(s2).float()
                    action = torch.from_numpy(a).float()
                    if not done or self.agent.conf.N_step == 0:
                        # add the step to memory, the last step add in pop with done=True
                        self.agent.memory.add((s_tensor,
                                               action,
                                               torch.tensor([r]).float(),
                                               s2_tensor,
                                               torch.tensor([self.agent.conf.gamma]),
                                               DATA_DEMO))
                        # Add new step to N-step and Pop N-step data to memory
                        if self.agent.conf.N_step > 0:
                            self.agent.backup.add_exp(
                                    (s_tensor, action, torch.tensor([r]).float(), s2_tensor))
                            self.agent.add_n_step_experience(DATA_DEMO, done)
            self.logger.info("{}/{} Demo Trajectories Loaded. Total Experience={}".format(
                demo_conf.load_N, demo_conf.demo_N, len(self.agent.memory)))
            self.agent.memory.set_protect_size(len(self.agent.memory))
        else:
            self.logger.info("No Demo Trajectory Loaded")


    def save_progress(self, display=False):
        # save model weight
        self.tp.save_model_weight(self.agent.actor_b, self.episode, prefix='actor_b')
        self.tp.save_model_weight(self.agent.actor_t, self.episode, prefix='actor_t')
        self.tp.save_model_weight(self.agent.critic_b, self.episode, prefix='critic_b')
        self.tp.save_model_weight(self.agent.critic_t, self.episode, prefix='critic_t')

        # save progress
        self.tp.save_progress(self.episode)
        self.tp.save_conf(self.conf.to_dict())
        if display:
            self.logger.info('Config name:' + self.conf.exp_name)
            self.logger.info('Progress Saved, current episode={}'.format(self.episode))


    def random_gen_dummy(self):
        batch_s = torch.randn(4096, 19)
        batch_a = torch.randn(4096, 7)
        batch_r = torch.randn(4096, 1)
        batch_s2 = torch.randn(4096, 19)
        batch_gamma = torch.randn(4096, 1)
        batch_flags = np.zeros((4096, ))
        weights = np.ones((4096,), dtype=float)
        idxes = list(range(4096))
        return batch_s, batch_a, batch_r, batch_s2, batch_gamma,\
               batch_flags, weights, idxes


    def update_agent(self, update_step):
        losses_critic = []
        losses_actor = []
        demo_cnt = []
        batch_sz = 0
        if self.agent.memory.ready():
            for _ in range(update_step):
                # Sample from memory
                t1 = time.time()
                (batch_s, batch_a, batch_r, batch_s2, batch_gamma, batch_flags), \
                weights, idxes = self.agent.memory.sample(self.conf.batch_size)
                # batch_s, batch_a, batch_r, batch_s2, batch_gamma, batch_flags, \
                # weights, idxes = self.random_gen_dummy()

                batch_s, batch_a, batch_r, batch_s2, batch_gamma, weights = \
                batch_s.to(self.device), batch_a.to(self.device), batch_r.to(self.device), \
                batch_s2.to(self.device), batch_gamma.to(self.device), \
                torch.from_numpy(weights.reshape(-1, 1)).float().to(self.device)

                # Get target network action and R, update every N' step
                t2 = time.time()
                batch_sz += batch_s.shape[0]
                with torch.no_grad():
                    action_tgt = self.agent.actor_t(batch_s)
                    y_tgt = batch_r + batch_gamma * self.agent.critic_t(
                            torch.cat((batch_s, action_tgt), dim=1))

                self.agent.zero_grad()
                # Critic loss
                t3 = time.time()
                self.optimizer_critic.zero_grad()
                Q_b = self.agent.critic_b(torch.cat((batch_s, batch_a), dim=1))
                loss_critic = (self.q_criterion(Q_b, y_tgt) * weights).mean()

                # Record Demo count
                t4 = time.time()
                d_flags = torch.from_numpy(batch_flags)
                demo_select = d_flags == DATA_DEMO
                N_act = demo_select.sum().item()
                demo_cnt.append(N_act)
                loss_critic.backward()
                self.optimizer_critic.step()

                # Actor loss
                t5 = time.time()
                self.optimizer_actor.zero_grad()
                action_b = self.agent.actor_b(batch_s)
                Q_act = self.agent.critic_b(torch.cat((batch_s, action_b), dim=1))
                loss_actor = -torch.mean(Q_act)
                loss_actor.backward()
                self.optimizer_actor.step()

                # Update priority
                t6 = time.time()
                priority = ((Q_b.detach() - y_tgt).pow(2) +\
                            Q_act.detach().pow(2)).cpu().numpy().ravel() +\
                            self.agent.conf.const_min_priority
                priority[batch_flags == DATA_DEMO] += self.agent.conf.const_demo_priority
                t6_5 = time.time()
                if not self.agent.conf.no_per:
                    self.agent.memory.update_priorities(idxes, priority)

                # Record loss
                t7 = time.time()
                losses_actor.append(loss_actor.item())
                losses_critic.append(loss_critic.item())
                # print(f"sample = {t2 - t1}s || target net = {t3 - t2}s || critic loss = {t4 - t3}s || loss backward = {t5 - t4}s || actor loss = {t6 - t5}s || priority = {t7 - t6}s")
                # print(f"priority = {t7-t6}s : phase1:{t6_5-t6}s | phase2:{t7-t6_5}s")

        if np.sum(demo_cnt) == 0:
            demo_n = 1e-10
        else:
            demo_n = np.sum(demo_cnt)
        return np.sum(losses_critic), np.sum(losses_actor), demo_n, batch_sz


    def summary(self):
        # call eval
        self.tp.add_meta(
                {'saved_episode': self.episode,
                 'np_random_state': np.random.get_state(),
                 'torch_random_state': torch.random.get_rng_state()})
        self.save_progress(display=True)


    def pretrain(self):
        assert self.full_conf.demo_config.load_demo_data

        self.agent.train()
        start_time = time.time()
        self.logger.info('Run Pretrain')

        for step in np.arange(self.conf.pretrain_save_step,
                              self.conf.pretrain_step + 1,
                              self.conf.pretrain_save_step):
            losses_critic, losses_actor, demo_n, batch_sz = \
                    self.update_agent(self.conf.pretrain_save_step)
            self.logger.info(
                    "{}-Pretrain Step {}/{}, (Mean): "\
                            "actor_loss={:.8f}, critic_loss={:.8f}, "\
                            "batch_sz = {}, Demo_ratio={:.8f}".format(
                                timeSince(start_time), step, self.conf.pretrain_step,
                                losses_actor / batch_sz, losses_critic / batch_sz,
                                batch_sz, demo_n / batch_sz))
            self.tp.record_step(step, 'pre_train',
                    {'actor_loss_mean': losses_actor / batch_sz,
                     'critic_loss_mean': losses_critic / batch_sz,
                     'batch_sz': batch_sz,
                     'Demo_ratio': demo_n / batch_sz}, display=False)
            self.episode = "pre_{}".format(step)
            self.summary()
            self.tp.plot_data("pre_train", self.conf.pretrain_save_step, step,
                    "result-pretrain-{}.png".format(self.episode),
                    self.conf.exp_name + str(self.conf.exp_idx) + '-Pretrain',
                    grid=False, ep_step=self.conf.pretrain_save_step)
        self.episode = 1


    def train(self):
        self.agent.train()

        start_time = time.time()
        while self.episode <= self.conf.n_episode:
            # episodic statistics
            eps_since = time.time()
            eps_reward = eps_length = eps_actor_loss = 0
            eps_critic_loss = eps_batch_sz = eps_demo_n = 0
            s0 = self.env.reset()

            self.agent.episode_reset()
            self.action_noise.reset()
            done = False
            s_tensor = self.agent.obs2tensor(s0)

            while not done:
                # 1.run env step
                with torch.no_grad():
                    t1 = time.time()
                    action_noise = torch.from_numpy(self.action_noise()).float()
                    t2 = time.time()
                    action = self.agent.actor_b(s_tensor.to(self.device)[None])[0].cpu() + action_noise
                    t3 = time.time()
                    s2, r, done, _ = self.env.step(action.numpy())
                    t4 = time.time()
                    # print(f"get noise time = {t2-t1}s\n",
                    #       f"get action time = {t3-t2}s\n",
                    #       f"step time = {t4-t3}s")
                    print(f"one-step cost = {t4-t1}s")
                    s2_tensor = self.agent.obs2tensor(s2)
                    # no-last step process
                    if not done or self.agent.conf.N_step == 0:
                        # add step to memory
                        self.agent.memory.add((s_tensor, action, torch.tensor([r]).float(),
                            s2_tensor, torch.tensor([self.agent.conf.gamma]),
                            DATA_RUNTIME))

                # 2.add new step to N-step and pos N-step data to memory
                us = time.time()
                if self.agent.conf.N_step > 0:
                    self.agent.backup.add_exp(
                            (s_tensor, action, torch.tensor([r]).float(), s2_tensor))
                    self.agent.add_n_step_experience(DATA_RUNTIME, done)

                losses_critic, losses_actor, demo_n, batch_sz = self.update_agent(self.conf.update_step)
                # losses_critic, losses_actor, demo_n, batch_sz = 1., 1., 1., 1.
                ue = time.time()
                print(f"update agent cost = {ue-us}s")
                print("====="*10)

                # 3.record episodic statistics
                eps_reward += r
                eps_length += 1
                eps_actor_loss += losses_actor
                eps_critic_loss += losses_critic
                eps_batch_sz += batch_sz
                eps_demo_n += demo_n

                # next step
                s_tensor = s2_tensor

            self.logger.info("{}: Episode {}-Last:{}: Actor_loss={:.8f}, Critic_loss={:.8f}, Step={}, Reward={}, Demo_ratio={:.8f}"\
                             .format(timeSince(start_time),
                                     self.episode,
                                     timeSince(eps_since),
                                     eps_actor_loss / eps_batch_sz,
                                     eps_critic_loss / eps_batch_sz,
                                     eps_length, eps_reward,
                                     eps_demo_n / eps_batch_sz))

            # Update target
            self.agent.update_target(self.agent.actor_b, self.agent.actor_t, self.episode)
            self.agent.update_target(self.agent.critic_b, self.agent.critic_t, self.episode)

            self.tp.record_step(self.episode, 'episode',
                                {'total_reward': eps_reward, 'length': eps_length,
                                 'avg_reward': eps_reward / eps_length,
                                 'elapsed_time': timeSince(eps_since, return_seconds=True),
                                 'actor_loss_mean': eps_actor_loss / eps_batch_sz,
                                 'critic_loss_mean': eps_critic_loss / eps_batch_sz,
                                 'eps_length': eps_length,
                                 'Demo_ratio': eps_demo_n / eps_batch_sz,
                                 }, display=False)
            if self.episode % self.conf.save_every == 0:
                self.eval()
                self.summary()
                self.tp.plot_data('episode', 1, self.episode, 'result-train-{}.png'.format(self.episode),
                        self.conf.exp_name+str(self.conf.exp_idx)+'-Episode', grid=False)
            self.episode += 1


    def eval(self, save_fig=True):
        self.agent.eval()
        all_length = []
        all_reward = []

        # backup env state
        for eps in range(self.conf.eval_episode):
            # episode statics
            eps_reward = eps_length = 0
            s0 = self.env.reset()
            done = False
            s_tensor = self.agent.obs2tensor(s0)

            while not done:
                with torch.no_grad():
                    action = self.agent.actor_b(s_tensor.to(self.device)[None])[0].cpu()
                    s2, r, done, _ = self.env.step(action.numpy())
                    s2_tensor = self.agent.obs2tensor(s2)

                eps_reward += r
                eps_length += 1

                # next step
                s_tensor = s2_tensor
            all_length.append(eps_length)
            all_reward.append(eps_reward)

        self.tp.record_step(self.episode, 'eval',
                            {'Mean Length': np.mean(all_length),
                             'Std Length': np.std(all_length),
                             'Mean Reward': np.mean(all_reward),
                             'Std Reward': np.std(all_reward)})
        self.logger.info('Eval Episode-{}: Mean Reward={:.3f}, Mean Length={:.3f}'\
                         .format(self.episode, np.mean(all_reward), np.mean(all_length)))

        if save_fig:
            self.tp.plot_data('eval', self.conf.save_every, self.episode,
                              'result-eval-{}.png'.format(self.episode),
                              self.conf.exp_name + str(self.conf.exp_idx) + '-Evaluate',
                              self.conf.save_every)
        #TODO: need set train?
        self.agent.train()


#TODO:
def analysis():
    import matplotlib.pyplot as plt

    def calc_ewma_reward(reward):
        reward_new = np.zeros(len(reward) + 1)
        #TODO Min reward of thr env
        reward_new[0] = -50
        ewma_reward = -50
        idx = 1
        for r in reward:
            ewma_reward = 0.05 * r + (1 - 0.05) * ewma_reward
            reward_new[idx] = ewma_reward
            idx += 1
        return reward_new

    pass

