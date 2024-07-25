import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG

import numpy as np
import sys
import gym_sumo
from gym_sumo.envs import SUMOEnv
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import wandb
from argparse import ArgumentParser
from utils.env_wrappers import DummyVecEnv, SubprocVecEnv
import time
import os
from tqdm import tqdm
import csv
from gym_sumo.envs.utils import generateFlowFiles
from gym_sumo.envs.utils import plot_scores
from gym_sumo.envs.utils import print_status
from copy import deepcopy

display = 'DISPLAY' in os.environ
use_gui = False
save = True
mode = 'gui' if (use_gui and display) else 'none'

# mode = 'gui'
run_mode = 'Train'
USE_CUDA = False  # torch.cuda.is_available()

EDGES = ['E0']
# EDGES = ['E0','-E1','-E2','-E3']
joint_agents = len(EDGES)>1
generateFlowFiles("Train", edges=EDGES)

env_kwargs = {'mode': mode,
              'edges': EDGES,
              'joint_agents': joint_agents}

class CustomVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        env = self.envs[0]
        self.buf_dones = np.zeros((self.num_envs, env.n), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs, env.n), dtype=np.float32)

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], a, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            self.buf_dones[env_idx] = a
            if all(self.buf_dones[env_idx]):
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action, joint_agents=False, load_state=False):
    def get_env_fn(rank):
        def init_env():
            env = SUMOEnv(mode=mode, edges=EDGES, joint_agents=joint_agents, load_state=load_state)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            # env.sumo_seed = seed + rank * 1000
            return env
        return init_env
    if n_rollout_threads == 1:
        return CustomVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config, wandb_run):
    model_dir = Path('./models') / config.env_id / config.model_name
    curr_run = f'maddpg_{4.87}_{config.seed}'
    if joint_agents:
        curr_run += '_joint'
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    # logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action, joint_agents=joint_agents, load_state=config.load_state)
    print(env.action_space)
    print(env.observation_space)
    
    env.env_method('set_run_mode', run_mode)
    # env.setInitialParameters(False)
    env.agent_types = env.get_attr('agent_types')[0]
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space.spaces.values()],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space.spaces])
    t = 0
    scores = []    
    smoothed_total_reward = 0
    pid = os.getpid()
    trainResultFilePath = f"stat_train_{pid}.csv"  
    with open(trainResultFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        written_headers = False

        for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
            total_reward = 0
            print("Episodes %i-%i of %i" % (ep_i + 1,
                                            ep_i + 1 + config.n_rollout_threads,
                                            config.n_episodes))
            obs = env.reset()
            print('first time flag', env.get_attr('firstTimeFlag'))
            step = 0
            # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
            maddpg.prep_rollouts(device='cpu')

            explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
            maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
            maddpg.reset_noise()
            # obs = 
            # oo = np.hstack(obs)
            # obs = [i[np.newaxis,:] for i in obs]
            for et_i in range(config.episode_length):
                step += 1
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[agentname])),
                                    requires_grad=False)
                            for agentname in env.get_attr('getAgentNames')[0]]
                # get actions as torch Variables
                torch_agent_actions = maddpg.step(torch_obs, explore=True)
                # convert actions to numpy arrays
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                # rearrange actions to be per environment
                actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                # env.envs[0].nextTimeSlot()
                simple_actions = simplify_actions(actions)
                # print(env.get_attr('edge_agents'))
                next_obs, rewards, dones, infos = env.step(simple_actions)
                replay_buffer.push(list(obs.values()), agent_actions, 
                                   rewards, list(next_obs.values()), dones)
                obs = next_obs
                t += config.n_rollout_threads
                total_reward += float(rewards[0][0])

                # rewardAgent_0, rewardAgent_1, rewardAgent_2 = env.env_method('rewardAnalysisStats')
                # print(env.env_method('rewardAnalysisStats'))

                # for edge_agent in env.get_attr('edge_agents'):
                #     print(edge_agent)
                #     headers, values = edge_agent.testAnalysisStats()
                #     if not written_headers:
                #         writer.writerow(headers + ['RewardAgent_0', 'RewardAgent_1', 'RewardAgent_2'])
                #         written_headers = True
                #     writer.writerow(values + [rewardAgent_0, rewardAgent_1, rewardAgent_2])

                val_losses = []
                pol_losses = []
                if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                    if USE_CUDA:
                        device = 'gpu'
                        maddpg.prep_training(device=device)
                    else:
                        device = 'cpu'
                        maddpg.prep_training(device=device)
                    for u_i in range(config.n_rollout_threads):
                        for a_i in range(maddpg.nagents):
                            sample = replay_buffer.sample(config.batch_size,
                                                        to_gpu=USE_CUDA)
                            val_loss, pol_loss = maddpg.update(sample, a_i)
                            val_losses.append(val_loss)
                            pol_losses.append(pol_loss)
                        maddpg.update_all_targets()
                    maddpg.prep_rollouts(device=device)
            ep_rews = replay_buffer.get_average_rewards(
                config.episode_length * config.n_rollout_threads)
            # for a_i, a_ep_rew in enumerate(ep_rews):
            #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
           
            total_reward /= step
            # show reward
            smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
            scores.append(smoothed_total_reward)
        
            # wandb.log({'# Episodes': ep_i, 
            #     "Average reward": round(np.mean(scores[-10:]), 2)})
            wandb_run.log({'# Episodes': ep_i, 
                "Average reward": smoothed_total_reward,
                'Actor loss': np.mean(pol_losses),
                'Critic loss': np.mean(val_losses)
                })

            if (ep_i % config.save_interval) < config.n_rollout_threads and save:
                os.makedirs(run_dir / 'incremental', exist_ok=True)
                maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
                maddpg.save(run_dir / 'model.pt')
        if save:
            maddpg.save(run_dir / 'model.pt')
        env.close()

    plt.plot(scores)
    plt.xlabel('episodes')
    plt.ylabel('ave rewards')
    plt.savefig('avgScore.jpg')
    # plt.show()
        # logger.export_scalars_to_json(str(log_dir / 'summary.json'))
        # logger.close()

def simplify_actions(actions):
    agent_actions = []
    for action in actions:
        if isinstance(action, list):
            agent_actions.append(simplify_actions(action))
        else:
            agent_actions.append(np.argmax(action))
    return agent_actions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple", type=str)
    parser.add_argument("--model_name", default="simple_model", type=str)
    parser.add_argument("--seed",
                        default=42, type=int,
                        help="Random seed") #42,43,44,45,46
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=4, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=1500, type=int)
    parser.add_argument("--episode_length", default=20, type=int)
    parser.add_argument("--steps_per_update", default=10, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=500, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=30, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument("--load_state", action='store_true')

    config = parser.parse_args()

    use_wandb = os.environ.get('WANDB_MODE', 'online') # can be online, offline, or disabled
    if not save:
        use_wandb = 'disabled'
    wandb_run = wandb.init(
        project=f"AdaptableLanesRevisionTRC{'MADDPG_'.lower()}",
        tags=["MADDPG_final?", "RL"],
        mode=use_wandb,
    )
    run(config, wandb_run)
