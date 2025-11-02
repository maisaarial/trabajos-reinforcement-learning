# optuna_practica.py
import os, time, random, json
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import ale_py

import optuna
from optuna.samplers import TPESampler

from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.triple_qlearning import TripleQLearning

# --- Wrapper: RAM -> estado discreto ---
class DiscreteHashObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_buckets: int = 50_000):
        super().__init__(env)
        self.n_buckets = int(n_buckets)
        self.observation_space = gym.spaces.Discrete(self.n_buckets)
    def observation(self, obs):
        import hashlib
        h = hashlib.sha1(np.asarray(obs, dtype=np.uint8).tobytes()).hexdigest()
        return int(h[:12], 16) % self.n_buckets

# ------------------------------------------------------------------
# Objetivo Optuna
# ------------------------------------------------------------------
def objective(trial: optuna.Trial):
    # --- Entorno Breakout idéntico a tu experimento ---
    gym.register_envs(ale_py)
    ENV_ID = "ALE/Breakout-v5"
    base_env = gym.make(ENV_ID, frameskip=1, full_action_space=False)
    base_env = TimeLimit(base_env, max_episode_steps=4000)
    env = DiscreteHashObsWrapper(base_env, n_buckets=50_000)

    seed = 47
    random.seed(seed); np.random.seed(seed)
    env.reset(seed=seed)

    # --- Escogemos algoritmo ---
    algo_name = trial.suggest_categorical("algo_name", ["sarsa","esarsa","qlearning","triple_q"])
    if   algo_name == "sarsa":     algo = Sarsa(env)
    elif algo_name == "esarsa":    algo = ExpectedSarsa(env)
    elif algo_name == "qlearning": algo = QLearning(env)
    else:                          algo = TripleQLearning(env)

    # --- Hiperparámetros ---
    n_steps = 100_000
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    lrdecay = trial.suggest_float("lrdecay", 0.90, 1.00, step=0.01)
    n_episodes_decay = trial.suggest_categorical("n_episodes_decay", [100, 1_000, 10_000])
    discount_rate = trial.suggest_float("discount_rate", 0.90, 1.00, step=0.01)

    # Política con *schedule* 
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5, step=0.1)
    exploration_initial_eps = 1.0
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.20, step=0.01)
    train_policy = EpsilonGreedyPolicy(
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        total_timesteps=n_steps
    )

    # --- Entrenamiento ---
    t0 = time.time()
    algo.learn(
        train_policy,
        n_steps=n_steps,
        discount_rate=discount_rate,
        lr=lr,
        lrdecay=lrdecay,
        n_episodes_decay=n_episodes_decay,
        tb_episode_period=500,
        verbose=False,
    )
    trial.set_user_attr("train_seconds", float(time.time() - t0))

    # --- Evaluación ---
    mean_reward, _ = evaluate_policy(env, algo.q_table, max_policy, n_episodes=100, verbose=False)

    env.close()
    return float(mean_reward)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    os.system("rm -rf ./logs/")
    os.makedirs("optuna", exist_ok=True)

    storage = "sqlite:///optuna/optuna.db"
    study_name = "breakout_tabular"
    sampler = TPESampler(seed=47)

    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    n_trials = 10
    print(f"Searching best hyperparameters in {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    outdir = f"optuna/{study_name}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/best_trial.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)

    # gráficas 
    try:
        optuna.visualization.plot_optimization_history(study).write_html(f"{outdir}/history.html")
        optuna.visualization.plot_param_importances(study).write_html(f"{outdir}/importances.html")
    except Exception:
        pass

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
