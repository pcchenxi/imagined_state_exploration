import gym

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.sac.policies import MlpPolicy

from policy import CustomPolicy 
from PPO_RC import PPO_RC

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds

# from stable_baselines import PPO2, SAC


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

env_id = "MountainCarContinuous-v0"
env = gym.make(env_id)
print(env.observation_space, env.action_space)
num_cpu = 1  # Number of processes to use
# Create the vectorized environment
env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

model = PPO_RC(CustomPolicy, env, n_steps=int(1024/num_cpu), ent_coef=0.0, nminibatches=64, noptepochs=4, \
                                    verbose=1, tensorboard_log='/home/xi/model')
model.learn(total_timesteps=int(5e4))

env_test = gym.make(env_id)
obs = env_test.reset()
for i in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env_test.step(action)
    env_test.render()
    if dones:
        env_test.reset()

# env_test = gym.make(env_id)
# # obs = env_test.reset()
# for i in range(100):
#     model.learn(total_timesteps=2000)

#     # model.save("ppo2_ipadgame")
#     # env.save_running_average("/home/xi/model/")
#     # print ('done')
#     obs = env_test.reset()
#     for i in range(1000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, rewards, dones, info = env_test.step(action)
#         env_test.render()
#         if dones:
#             env_test.reset()