import os
import time
import random
import optuna
import gymnasium as gym
import numpy as np

from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.triple_qlearning import TripleQLearning  # <-- tu nuevo algoritmo

# ============================================================
#  Wrapper: Discretiza la observación RAM de Breakout a N buckets
#  (hash estable -> entero [0..n_buckets-1]) para tabular.
# ============================================================
class DiscreteHashObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_buckets: int = 50000):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), \
            "Se espera observación tipo Box (RAM/pixels)."
        self.n_buckets = int(n_buckets)
        self.observation_space = gym.spaces.Discrete(self.n_buckets)

    def observation(self, obs):
        # obs es np.ndarray (RAM de 128 bytes en Breakout-ram)
        # Usamos SHA1 para hash estable y reducimos mod n_buckets
        import hashlib
        h = hashlib.sha1(obs.tobytes()).hexdigest()
        idx = int(h[:12], 16) % self.n_buckets
        return idx


# ============================================================
#  Entrenamiento + evaluación (patrón activity_2.8b.py)
# ============================================================
def train_and_evaluate(algo, n_steps=200_000, epsilon=0.1,
                       discount_rate=0.99, lr=0.1, lrdecay=1.0,
                       n_episodes_decay=10_000, tb_episode_period=500):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    start_time = time.time()
    algo.learn(
        epsilon_greedy_policy,
        n_steps=n_steps,
        discount_rate=discount_rate,
        lr=lr,
        lrdecay=lrdecay,
        n_episodes_decay=n_episodes_decay,
        tb_episode_period=tb_episode_period,
        verbose=False
    )
    dur = time.time() - start_time
    mean_r, std_r = evaluate_policy(
        algo.env, algo.q_table, max_policy, n_episodes=10, verbose=False
    )
    return mean_r, std_r, dur


# ============================================================
#  Objetivo Optuna por algoritmo
# ============================================================
def make_objective(env_name, algo_class, hash_buckets, n_steps):
    def objective(trial: optuna.Trial):
        # Hiperparámetros a explorar
        epsilon = trial.suggest_float("epsilon", 0.05, 0.4, step=0.05)
        discount_rate = trial.suggest_float("discount_rate", 0.90, 0.999)
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        lrdecay = trial.suggest_float("lrdecay", 0.90, 1.0)
        n_episodes_decay = trial.suggest_int("n_episodes_decay", 2_000, 20_000, step=2_000)

        # Entornos
        base_env = gym.make(env_name, frameskip=1, full_action_space=False)
        env = DiscreteHashObsWrapper(base_env, n_buckets=hash_buckets)

        # Semilla
        seed = 3
        random.seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)

        # Algoritmo
        algo = algo_class(env)

        # Entrenamiento + evaluación
        mean_r, std_r, dur = train_and_evaluate(
            algo,
            n_steps=n_steps,
            epsilon=epsilon,
            discount_rate=discount_rate,
            lr=lr,
            lrdecay=lrdecay,
            n_episodes_decay=n_episodes_decay,
            tb_episode_period=1_000
        )

        # Reporte intermedio (para pruners si se añaden)
        trial.set_user_attr("std_reward", float(std_r))
        trial.set_user_attr("duration_sec", float(dur))
        return float(mean_r)
    return objective


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    # Limpia logs de TB para una sesión nueva (igual que tu plantilla)
    os.system("rm -rf ./logs/")

    # --------- Configuración del estudio ---------
    # Usamos la versión RAM para poder tabular (discretización por hash).
    ENV_NAME = "ALE/Breakout-ram-v5"

    # Buckets del wrapper (trade-off estado/tabla)
    HASH_BUCKETS = 50_000

    # Pasos de entrenamiento por trial (ajústalo según recursos)
    N_STEPS = 400_000

    # Número de trials por algoritmo
    N_TRIALS = 10

    # DB de Optuna (persistencia)
    os.makedirs("optuna", exist_ok=True)
    storage = "sqlite:///optuna/optuna.db"

    # --------- Lista de algoritmos a comparar ---------
    algos = [
        ("SARSA", Sarsa),
        ("ExpectedSARSA", ExpectedSarsa),
        ("QLearning", QLearning),
        ("TripleQLearning", TripleQLearning),  # tu nuevo algoritmo
    ]

    best_summary = []

    for study_name, algo_class in algos:
        print(f"\n=== Optuna Study: {study_name} ===")
        study = optuna.create_study(
            study_name=f"Breakout_{study_name}",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(
            make_objective(ENV_NAME, algo_class, HASH_BUCKETS, N_STEPS),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

        print(f"[{study_name}] Best value (mean_reward): {study.best_value:.4f}")
        print(f"[{study_name}] Best params: {study.best_params}")
        best_summary.append((study_name, study.best_value, study.best_params))

    print("\n==================== RESUMEN FINAL ====================")
    for name, val, params in best_summary:
        print(f"{name:18s}  mean_reward={val:8.4f}   params={params}")

    print("\nBase de datos Optuna: optuna/optuna.db")
    print("Puedes lanzar el dashboard con:")
    print("  optuna-dashboard sqlite:///optuna/optuna.db")
