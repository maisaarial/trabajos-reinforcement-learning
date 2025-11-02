import os
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import ale_py

from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.triple_qlearning import TripleQLearning

# --- Wrapper mínimo para discretizar la RAM (128 bytes) a un entero ---
class DiscreteHashObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_buckets: int = 50_000):
        super().__init__(env)
        self.n_buckets = int(n_buckets)
        self.observation_space = gym.spaces.Discrete(self.n_buckets)

    def observation(self, obs):
        import hashlib
        h = hashlib.sha1(np.asarray(obs, dtype=np.uint8).tobytes()).hexdigest()
        return int(h[:12], 16) % self.n_buckets


def train_and_evaluate(algo, n_steps, policy_kwargs, **learn_kwargs):
    """
    Entrena con una política ε-greedy programada (schedule) y luego evalúa
    con política codiciosa (max_policy). Devuelve (mean_reward, std_reward).
    """
    eps_policy = EpsilonGreedyPolicy(
        exploration_fraction=policy_kwargs["exploration_fraction"],
        exploration_initial_eps=1.0,
        exploration_final_eps=policy_kwargs["exploration_final_eps"],
        total_timesteps=n_steps
    )
    t0 = time.time()
    algo.learn(eps_policy, n_steps, **learn_kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - t0))

    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)


if __name__ == "__main__":
    os.system("rm -rf ./logs/")

    # Registrar ALE y crear entornos 
    gym.register_envs(ale_py)
    ENV_ID = "ALE/Breakout-v5"

    base_env = gym.make(ENV_ID, frameskip=1, full_action_space=False)
    base_env = TimeLimit(base_env, max_episode_steps=4000)
    env = DiscreteHashObsWrapper(base_env, n_buckets=50_000)

    base_visual = gym.make(ENV_ID, render_mode='rgb_array', frameskip=1, full_action_space=False)
    base_visual = TimeLimit(base_visual, max_episode_steps=4000)
    visual_env = DiscreteHashObsWrapper(base_visual, n_buckets=50_000)

    # Semilla
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

    # Pasos de entrenamiento 
    N_STEPS_LONG = 300_000 

    resultados = []

    # =========================
    #  SARSA  
    # =========================
    sarsa = Sarsa(env)
    print("Entrenando SARSA (mejores hiperparámetros)")
    mean_sr, std_sr = train_and_evaluate(
        sarsa,
        n_steps=N_STEPS_LONG,
        policy_kwargs=dict(exploration_fraction=0.30, exploration_final_eps=0.20),
        discount_rate=0.94,
        lr=0.0124,
        lrdecay=0.92,
        n_episodes_decay=1000,
        tb_episode_period=500,
        verbose=False
    )
    evaluate_policy(visual_env, sarsa.q_table, max_policy, n_episodes=10, verbose=False)
    resultados.append(("SARSA", mean_sr, std_sr))

    # =========================
    #  Triple Q-Learning
    # =========================
    tql = TripleQLearning(env)
    print("Entrenando Triple Q-Learning (mejores hiperparámetros)")
    mean_tq, std_tq = train_and_evaluate(
        tql,
        n_steps=N_STEPS_LONG,
        policy_kwargs=dict(exploration_fraction=0.40, exploration_final_eps=0.19),
        discount_rate=0.99,
        lr=0.0106,
        lrdecay=0.91,
        n_episodes_decay=100,
        tb_episode_period=500,
        verbose=False
    )
    evaluate_policy(visual_env, tql.q_table, max_policy, n_episodes=10, verbose=False)
    resultados.append(("TripleQLearning", mean_tq, std_tq))

    # =========================
    #  Expected SARSA
    # =========================
    esarsa = ExpectedSarsa(env)
    print("Entrenando Expected SARSA (mejores hiperparámetros)")
    mean_es, std_es = train_and_evaluate(
        esarsa,
        n_steps=N_STEPS_LONG,
        policy_kwargs=dict(exploration_fraction=0.20, exploration_final_eps=0.01),
        discount_rate=0.92,
        lr=0.0260,
        lrdecay=0.98,
        n_episodes_decay=10_000,
        tb_episode_period=500,
        verbose=False
    )
    evaluate_policy(visual_env, esarsa.q_table, max_policy, n_episodes=10, verbose=False)
    resultados.append(("ExpectedSARSA", mean_es, std_es))

    # =========================
    #  Q-Learning
    # =========================
    ql = QLearning(env)
    print("Entrenando Q-Learning (mejores hiperparámetros)")
    mean_ql, std_ql = train_and_evaluate(
        ql,
        n_steps=N_STEPS_LONG,
        policy_kwargs=dict(exploration_fraction=0.50, exploration_final_eps=0.16),
        discount_rate=0.96,
        lr=0.0076,
        lrdecay=0.95,
        n_episodes_decay=10_000,
        tb_episode_period=500,
        verbose=False
    )
    evaluate_policy(visual_env, ql.q_table, max_policy, n_episodes=10, verbose=False)
    resultados.append(("QLearning", mean_ql, std_ql))

    # === Resumen final===
    print("\n===== RESUMEN ENTRENAMIENTO LARGO (100 episodios de evaluación) =====")
    for name, m, s in resultados:
        print(f"{name:16s} -> mean_reward = {m:.3f} | std = {s:.3f}")
