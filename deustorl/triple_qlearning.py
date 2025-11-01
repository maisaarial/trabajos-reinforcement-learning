from deustorl.common import *
from deustorl.helpers import TensorboardLogger
import random

class TripleQLearning:
    """
    Triple Q-Learning:
      - Mantiene tres tablas Q (Q1, Q2, Q3).
      - En cada paso elige aleatoriamente UNA tabla para actualizar.
      - Para el bootstrap (max) usa la acción codiciosa de la tabla elegida y
        evalúa su valor como el promedio de las OTRAS dos tablas (estimator ensemble).
      - La política de actuación se basa en la Q media (Q̄ = (Q1+Q2+Q3)/3), expuesta
        como self.q_table para mantener compatibilidad con el resto del código.
    """
    def __init__(self, env):
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        # Tres estimadores
        self.q1 = QTable(n_states, n_actions)
        self.q2 = QTable(n_states, n_actions)
        self.q3 = QTable(n_states, n_actions)
        # Vista promedio (para actuar con la policy y para inspección)
        self.q_table = QTable(n_states, n_actions)
        self._sync_mean_table_full()
        self.env = env

    # ---------- Utils ----------
    def _sync_mean_table_state(self, s: int):
        """Recalcula la fila promedio sólo para el estado s."""
        for a in range(len(self.q_table[s])):
            self.q_table[s][a] = (self.q1[s][a] + self.q2[s][a] + self.q3[s][a]) / 3.0

    def _sync_mean_table_full(self):
        """Recalcula la tabla promedio completa (se usa en __init__)."""
        for s in range(self.q_table.n_states):
            self._sync_mean_table_state(s)

    # ---------- Learning ----------
    def learn(self, policy, n_steps: int = 100, discount_rate=1.0, lr=0.01, lrdecay=1.0,
              n_episodes_decay=100, tb_episode_period=100, verbose=False):
        
        obs, _ = self.env.reset()
        selected_action = policy(self.q_table[obs])

        tblogger = TensorboardLogger(
            "TripleQLearning_(dr=" + str(discount_rate)
            + "-lr=" + str(lr)
            + "-lrdecay=" + str(lrdecay)
            + "e" + str(n_episodes_decay) + ")",
            episode_period=tb_episode_period
        )

        n_episodes = 0
        episode_reward = 0.0
        episode_steps = 0

        for _ in range(n_steps):
            prev_obs = obs
            prev_action = selected_action

            obs, reward, terminated, truncated, _ = self.env.step(selected_action)

            episode_reward += reward
            episode_steps += 1

            if verbose:
                self.env.render()

            # Acción siguiente según la política sobre la Q promedio
            selected_action = policy(self.q_table[obs])

            # --- Actualización Triple Q-Learning ---
            # Elegimos aleatoriamente cuál estimador actualizar
            which = random.randint(1, 3)
            if which == 1:
                q_sel, q_e1, q_e2 = self.q1, self.q2, self.q3
            elif which == 2:
                q_sel, q_e1, q_e2 = self.q2, self.q1, self.q3
            else:
                q_sel, q_e1, q_e2 = self.q3, self.q1, self.q2

            # Selección greedy en el estado siguiente usando la tabla elegida
            # (análogamente a Double Q-Learning)
            next_row_sel = q_sel[obs]
            a_star = max(range(len(next_row_sel)), key=lambda a: next_row_sel[a])

            # Valor bootstrap: media de las otras dos tablas en a*
            max_estimate = (q_e1[obs][a_star] + q_e2[obs][a_star]) / 2.0

            # Objetivo TD (Q-Learning)
            td_target = reward + discount_rate * max_estimate

            # Update en la tabla elegida
            q_sel[prev_obs][prev_action] += lr * (td_target - q_sel[prev_obs][prev_action])

            # Mantener sincronizada la Q promedio (al menos para el estado actualizado)
            self._sync_mean_table_state(prev_obs)

            # --- Fin de episodio ---
            if terminated or truncated:
                tblogger.log(episode_reward, episode_steps)
                if verbose:
                    print(self.q_table)
                    print("--- EPISODE STARTS ---")

                episode_reward = 0.0
                episode_steps = 0
                obs, _ = self.env.reset()
                selected_action = policy(self.q_table[obs])

                # lr decay por episodios
                n_episodes += 1
                if n_episodes % n_episodes_decay == 0:
                    lr *= lrdecay

