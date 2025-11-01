import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, NoopResetEnv, FireResetEnv
import ale_py  
import random
import time
import numpy as np

from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.triple_qlearning import TripleQLearning

# --- Wrapper mínimo para discretizar la RAM (128 bytes) a un entero ---
class DiscreteHashObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_buckets: int = 50000):
        super().__init__(env)
        self.n_buckets = int(n_buckets)
        self.observation_space = gym.spaces.Discrete(self.n_buckets)

    def observation(self, obs):
        import hashlib
        h = hashlib.sha1(np.asarray(obs, dtype=np.uint8).tobytes()).hexdigest()
        return int(h[:12], 16) % self.n_buckets

def train_and_evaluate(algo, n_steps=60000, **kwargs):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)

os.system("rm -rf ./logs/")

# registra ALE en Gymnasium
gym.register_envs(ale_py)

# Entorno Breakout versión RAM (observación Box(128,)), discretizado para tabular
env_name = "ALE/Breakout-v5"
base_env = gym.make(env_name, frameskip=1, full_action_space=False)
# para solucionar ese error del limit
base_env = NoopResetEnv(base_env, noop_max=30)
base_env = FireResetEnv(base_env)
base_env = TimeLimit(base_env, max_episode_steps=4000) 

env = DiscreteHashObsWrapper(base_env, n_buckets=50000)


base_visual_env = gym.make(env_name, render_mode='human', frameskip=1, full_action_space=False)
base_visual_env = NoopResetEnv(base_visual_env, noop_max=30)
base_visual_env = FireResetEnv(base_visual_env)
base_visual_env = TimeLimit(base_visual_env, max_episode_steps=4000)  # 👈 límite duro
visual_env = DiscreteHashObsWrapper(base_visual_env, n_buckets=50000)

# Visualización (mismo patrón: env con render_mode='human')

seed = 3
random.seed(seed)
env.reset(seed=seed)

n_steps = 200_000

algo = TripleQLearning(env)
print("Testing Triple Q-Learning")
train_and_evaluate(algo, n_steps=n_steps, lr=0.1)
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=10, verbose=False)

algo = Sarsa(env)
print("Testing SARSA")
train_and_evaluate(algo, n_steps=n_steps, lr=0.1)
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=10, verbose=False)

algo = QLearning(env)
print("Testing Q-Learning")
train_and_evaluate(algo, n_steps=n_steps, lr=0.1)
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=10, verbose=False)

algo = ExpectedSarsa(env)
print("Testing Expected SARSA")
train_and_evaluate(algo, n_steps=n_steps, lr=0.1)
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=10, verbose=False)

