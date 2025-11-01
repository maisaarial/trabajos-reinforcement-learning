from deustorl.common import *
from deustorl.helpers import TensorboardLogger

class ExpectedSarsa:
    def __init__(self, env):
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q_table = QTable(n_states, n_actions)
        self.env = env

    def learn(self, policy, n_steps:int=100, discount_rate=1, lr=0.01, lrdecay=1.0, n_episodes_decay=100, tb_episode_period=100, verbose=False):
        obs,_ = self.env.reset()
        selected_action = policy(self.q_table[obs])

        tblogger = TensorboardLogger("ExpectedSARSA_(dr=" + str(discount_rate) +"-lr=" + str(lr) + "-lrdecay=" + str(lrdecay) + "e"+ str(n_episodes_decay) + ")" , episode_period=tb_episode_period)

        n_episodes = 0
        episode_reward = 0
        episode_steps = 0

        for n in range(n_steps):
            previous_obs = obs
            previous_action = selected_action
            obs, reward, terminated, truncated, _ = self.env.step(selected_action)

            episode_reward += reward
            episode_steps += 1

            if verbose:
                self.env.render()

            selected_action = policy(self.q_table[obs])

            average_qvalue = self.calculate_average_qvalue(self.q_table[obs], epsilon=policy.epsilon)

            td = reward + discount_rate*average_qvalue

            self.q_table[previous_obs][previous_action] += lr*(td-self.q_table[previous_obs][previous_action])

            if terminated or truncated:
                tblogger.log(episode_reward, episode_steps)
                if verbose:
                    print(self.q_table)
                # reset the environment and reinitialize trajectory
                if verbose:
                    print("--- EPISODE STARTS ---")
                episode_reward = 0
                episode_steps = 0
                obs,_ = self.env.reset()
                selected_action = policy(self.q_table[obs])
                
                # lrdacay update
                n_episodes += 1
                if n_episodes % n_episodes_decay == 0:
                    lr *= lrdecay

    def calculate_average_qvalue(self, values, epsilon=0):
        max_value = max (values)
        n_actions = len(values)
        n_greedy_actions = sum(1 for v in values if v == max_value)

        non_greedy_action_probability = epsilon / n_actions
        greedy_action_probability = ((1 - epsilon) / n_greedy_actions) + non_greedy_action_probability
 
        result = 0
        for v in values:
            if v == max_value:
                result += v * greedy_action_probability
            else:
                result += v * non_greedy_action_probability

        return result
   