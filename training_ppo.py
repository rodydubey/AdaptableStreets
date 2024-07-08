import numpy as np
import gym_sumo
import gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb
import argparse
import os
from pathlib import Path

from gym_sumo.envs import SUMOEnv
from gym_sumo.envs.utils import generateFlowFiles
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')



class SUMOEnvPPO(SUMOEnv):
    def __init__(self, reset_callback=None, reward_callback=None, observation_callback=None, info_callback=None, done_callback=None, shared_viewer=True, mode='gui', 
                 edges=..., simulation_end=36000, joint_agents=False, episode_length=20, **kwargs):
        super().__init__(reset_callback, reward_callback, observation_callback, info_callback,
                         done_callback, shared_viewer, mode, edges, simulation_end, joint_agents, **kwargs)
        self.action_space = gym.spaces.MultiDiscrete([5, 9, 2])
        # observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(11,), dtype=np.float64)
        self.episode_length = episode_length
        self._episode_length_counter = 0

    def _get_obs(self, agent):
        return self.getState(self.edge_agents[0])

    def getState(self, edge_agent):
        """
        Retrieve the state of the network from sumo. 
        """
        edge_id = edge_agent.edge_id
        normalizeUniqueVehicleCount = 300
        laneWidthCar = self.traci.lane.getWidth(f'{edge_id}_2')
        laneWidthBike = self.traci.lane.getWidth(f'{edge_id}_1')
        laneWidthPed = self.traci.lane.getWidth(f'{edge_id}_0')
        nLaneWidthCar = np.interp(laneWidthCar, [0, 12.6], [0, 1])
        nLaneWidthBike = np.interp(laneWidthBike, [0, 12.6], [0, 1])
        nLaneWidthPed = np.interp(laneWidthPed, [0, 12.6], [0, 1])

        # E0 is for agent 0 and 1, #-E0 is for agent 2 and 3, #E1 is for agent 4 and 5, #-E1 is for agent 6 and 7
        # E2 is for agent 8 and 9, #-E2 is for agent 10 and 11, #E3 is for agent 12 and 13, #-E3 is for agent 14 and 15

        laneVehicleAllowedType = self.traci.lane.getAllowed(f'{edge_id}_0')
        if 'bicycle' in laneVehicleAllowedType:
            cosharing = 1
        else:
            cosharing = 0

        state = []
        state_0 = laneWidthCar
        state_1 = laneWidthBike
        state_2 = laneWidthPed
        state_3 = edge_agent._total_occupancy_car_Lane
        state_4 = edge_agent._total_density_car_lane
        state_5 = edge_agent._total_occupancy_bike_Lane
        state_6 = edge_agent._total_occupancy_ped_Lane
        state_7 = float(cosharing)  # flag for cosharing on or off
        state_8 = float(np.abs(cosharing-1))
        state_9 = edge_agent._total_density_bike_lane
        state_10 = edge_agent._total_density_ped_lane

        state = [state_0, state_1, state_2, state_3, state_4,
                 state_5, state_6, state_7, state_8, state_9, state_10]

        return np.array(state)

    def reset(self, *args):
        obs = super().reset(*args)
        self._episode_length_counter = 0
        return list(obs.values())[0]

    def step(self, action_n):
        obs_n, reward_n, done_n, info_n = super().step(action_n)
        self._episode_length_counter += 1
        obs_n = list(obs_n.values())[0]
        reward_n = reward_n[0]

        # if self._episode_length_counter >= 20:
        # 	self._done = True
        done_n = self._episode_length_counter >= self.episode_length
        return obs_n, reward_n, done_n, info_n



def run(model, config):
    model_dir = Path('./models') / config.env_id / config.model_name
    curr_run = f'ppo_{env.get_attr("density_threshold")[0]:.2f}_{config.seed}'
    #   if not model_dir.exists():
    #       curr_run = 'run1'
    #   else:
    #     exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
    #                       model_dir.iterdir() if
    #                       str(folder.name).startswith('run')]
    #     if len(exst_run_nums) == 0:
    #         curr_run = 'run1'
    #     else:
    #         curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    # for ep_i in tqdm(range(0, config.n_episodes)):
    #   total_reward = 0
    #   print("Episodes %i-%i of %i" % (ep_i + 1,
    #                                           ep_i + 1 + config.n_rollout_threads,
    #                                           config.n_episodes))
    #   # obs = env.reset()
    # Train the agent
    model.learn(total_timesteps=config.n_episodes*config.episode_length, reset_num_timesteps=True, tb_log_name="PPO", progress_bar=True,
                callback=WandbCallback(
                    model_save_path=run_dir,
                    verbose=2,))
    # step = 0
    # for et_i in range(config.episode_length):
    #   step += 1
    #   action, _ = model.predict(obs, deterministic=True)
    #   print("Step {}".format(step + 1))
    #   print("Action: ", action)
    #   obs, reward, done, info = env.step(action)
    #   # model.learn(1)
    #   print('obs=', obs, 'reward=', reward, 'done=', done)
    #   if done:
    #     print("Goal reached!", "reward=", reward)
    #     break
    model.save(run_dir / 'model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple", type=str)
    parser.add_argument("--model_name", default="simple_model", type=str)
    parser.add_argument("--seed",
                        default=42, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=1500, type=int)
    parser.add_argument("--episode_length", default=20, type=int)
    parser.add_argument("--steps_per_update", default=10, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=24000, type=int)
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

    config = parser.parse_args()


    EDGES = ['E0']
    generateFlowFiles("Train", edges=EDGES)
    joint_agents = False
    mode = 'none'
    env_kwargs = {'mode': mode,
                'edges': EDGES,
                'joint_agents': joint_agents}

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # can be online, offline, or disabled
    use_wandb = os.environ.get('WANDB_MODE', 'online')
    wandb_run = wandb.init(
        project=f"AdaptableLanesRevisionTRC{'PPO_'.lower()}",
        tags=["PPO_Final?", "RL"],
        mode=use_wandb,
        sync_tensorboard=True
    )

    env_kwargs['episode_length'] = config.episode_length

    env = make_vec_env(SUMOEnvPPO, n_envs=4,
                       seed=config.seed, env_kwargs=env_kwargs)
    env.env_method('set_run_mode', 'Train')
    
    # new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    # check_env(env, warn=True)
    print(env.action_space)
    print(env.action_space.sample())
    print(env.observation_space)
    # env.reset()
    model = PPO("MlpPolicy", env, n_steps=20, verbose=1,
                tensorboard_log=f'logs/{wandb_run.id}', device='cpu', seed=config.seed)
    # model.set_logger(new_logger)
    # print(f'logs/{wandb_run.id}')
    run(model, config)
    wandb_run.finish()
