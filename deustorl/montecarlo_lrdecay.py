import numpy as np
from deustorl.common import QTable, ReturnsTable
from deustorl.helpers import TensorboardLogger

class Montecarlo_FirstVisit_LRDecay:
    def __init__(self, env):
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q_table = QTable(n_states, n_actions)
        self.env = env

    def learn(self, policy, n_steps:int=100, discount_rate=1, lr=0.1, lrdecay=0.99, n_episodes_decay=100, tb_episode_period=100, verbose=False):
        obs,_ = self.env.reset()
        selected_action = policy(self.q_table[obs])
        trajectory = []

        tblogger = TensorboardLogger("MonteCarloLR_(dr=" + str(discount_rate) +"-lr=" + str(lr) +")" , episode_period=tb_episode_period)
        
        n_episodes = 0 # For the LR decay functionality

        for n in range(n_steps):
            old_obs = obs
            obs, reward, terminated, truncated, _ = self.env.step(selected_action)

            trajectory.append({"state":old_obs, "action":selected_action, "reward":reward})
            if verbose:
                print(self.env.render())

            selected_action = policy(self.q_table[obs])

            if terminated or truncated:
                # analyze episode
                G = 0
                for i, step in enumerate(reversed(trajectory)):
                    state = step["state"]
                    action = step["action"]
                    reward = step["reward"]
                    G = G*discount_rate + reward

                    if i == 0:
                        # Calculate the accumulated reward for logging purposes
                        reward = 0
                        for _, element in enumerate(trajectory):
                            reward += element["reward"]
                        tblogger.log(reward,len(trajectory))

                    if not self.visited_state_action(state, action, trajectory, i+1):
                        self.q_table[state][action] += lr * (G - self.q_table[state][action])

                if verbose:
                    print(self.q_table)
                # reset the environment and reinitialize trajectory
                if verbose:
                    print("--- EPISODE STARTS ---")
                obs,_ = self.env.reset()
                selected_action = policy(self.q_table[obs])
                trajectory = []
                G = 0

                # LR decay logic
                n_episodes += 1
                if n_episodes % n_episodes_decay == 0:
                    lr *= lrdecay


    def visited_state_action(self, state, action, trajectory, max) -> bool:
        for step in trajectory[:-max]:
            if step["state"]== state and step["action"]==action:
                return True
        return False 
