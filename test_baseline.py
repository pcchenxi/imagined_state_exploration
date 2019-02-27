#!/usr/bin/env python3
import gym

from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
# from ppo_morl.ppo2 import PPO_MORL
from stable_baselines.ppo2 import PPO2
from stable_baselines.common import set_global_seeds


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 64, 32],
                                                          vf=[128, 64, 32])],
                                           feature_extraction="mlp")

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

env_id = 'Pendulum-v0'
env = gym.make(env_id)
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
env = SubprocVecEnv([make_env(env_id, i) for i in range(128)])
env = VecNormalize(env)

model = PPO2(CustomPolicy, env, n_steps=int(2048/128), nminibatches=64, noptepochs=10, lam=0.98, verbose=1, tensorboard_log='/home/xi/model/log')
# model = PPO2.load("ppo2_ipadgame")
# model.set_env(env)
# model.tensorboard_log='/home/xi/model/log'
# env.load_running_average("/home/xi/model/")

model.learn(total_timesteps=50000)

# model.save("ppo2_ipadgame")
# env.save_running_average("/home/xi/model/")
# print ('done')

env = gym.make(env_id)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env)
obs = env.reset()
model = PPO2.load("ppo2_ipadgame")
env.load_running_average("/home/xi/model/")

for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()