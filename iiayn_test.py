import gym

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.sac.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds

from iiayn.sac import SAC
from iiayn.policies import MlpPolicy

# from iiayn.ppo2 import PPO2
# from policy import CustomPolicy 

from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv


def make_env(env_id, rank, seed=0):
    def _init():
        # env = gym.make(env_id)
        # env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

# env_id = "MountainCarContinuous-v0"
# env = gym.make(env_id)
env = PointMazeEnv()
num_cpu = 1  # Number of processes to use

# env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
env = DummyVecEnv([lambda: env])

model = SAC(MlpPolicy, env, verbose=1, tensorboard_log='/home/xi/model')
# model.learn(total_timesteps=int(5e5))

# model = PPO2(CustomPolicy, env, n_steps=int(1024/num_cpu), ent_coef=0.0, nminibatches=64, noptepochs=4, \
#                                     verbose=1, tensorboard_log='/home/xi/model')
model.learn(total_timesteps=int(5e5))

env_test = gym.make(env_id)
obs = env_test.reset()
for i in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env_test.step(action)
    env_test.render()
    if dones:
        env_test.reset()
