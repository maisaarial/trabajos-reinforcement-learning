from collections import deque
import datetime
from statistics import mean
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from torch.utils.tensorboard.writer import SummaryWriter


class FrozenLakeDenseRewardsWrapper(gym.Wrapper):
    """
    A wrapper class for the FrozenLake environment that provides dense rewards based on the distance to the goal state.
    Args:
        env (gym.Env): The original FrozenLake environment.
    Attributes:
        env (gym.Env): The original FrozenLake environment.
        rows (int): The number of rows in the FrozenLake grid.
        columns (int): The number of columns in the FrozenLake grid.
        min_distance (int): The minimum distance from the current state to the goal state.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.rows = self.env.unwrapped.nrow
        self.columns = self.env.unwrapped.ncol

    def reset(self, seed = None, options = None):
        """
        Overrides the reset method of the parent class to reset the minimum distance to the maximum possible distance.
        """
        self.min_distance = self.env.unwrapped.nrow + self.env.unwrapped.ncol
        return super().reset(seed=seed, options=options)

    def step(self, action):
        """
        Overrides the step method of the parent class to calculate the distance to the goal state and modify the reward accordingly.
        Ir only provides reward if the new distance is less than the minimum distance found so far.
        Args:
            action (int): The action to take in the environment.
        Returns:
            tuple: A tuple containing the next state, the modified reward, a flag indicating if the episode is terminated, a flag indicating if the episode is truncated, and additional information.
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if not terminated:
            state_row = next_state // (self.rows - 1)
            state_column =  next_state % (self.columns - 1)
            distance = self.rows - 1 - state_row + self.columns - 1 - state_column
            if distance < self.min_distance:
                self.min_distance = distance
                reward += 0.01
        return next_state, reward, terminated, truncated, info
    
    
class TensorboardLogger():
    def __init__(self, algo_name="algo", episode_period=1, log_dir="logs"):
        self.log_dir = log_dir
        self.tb_writer = SummaryWriter(self.log_dir + "/" + algo_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.episode_rewards = deque([], maxlen=episode_period)
        self.episode_steps = deque([], maxlen=episode_period)
        self.total_steps = 0
        
    def log(self, episode_reward, episode_steps):
        self.total_steps += episode_steps

        self.episode_rewards.append(episode_reward)
        self.episode_steps.append(episode_steps)
        average_rewards = mean(self.episode_rewards)
        average_steps = mean(self.episode_steps)


        self.tb_writer.add_scalar("train/reward", average_rewards, self.total_steps)
        self.tb_writer.add_scalar("train/episode_len", average_steps, self.total_steps) 


class DiscretizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box), "The observation space must be a Box space."

        low = self.observation_space.low if low is None else np.array(low)
        high = self.observation_space.high if high is None else np.array(high)

        # Clip values in low and high to be within the specified bounds
        max_abs_value = 1e6
        low = np.clip(low, -max_abs_value, max_abs_value)
        high = np.clip(high, -max_abs_value, max_abs_value)

        # Calculate the width of each bin
        self.bin_widths = (high - low) / n_bins

        self.n_bins = n_bins
        self.low = low
        self.high = high
        self.ob_shape = self.observation_space.shape

        # Define the new discrete observation space
        self.observation_space = Discrete(n_bins ** low.size)
        print("New observation space:", self.observation_space)

    def _discretize_observation(self, observation):
        # Clip the observation to ensure it stays within the adjusted bounds
        clipped_obs = np.clip(observation, self.low, self.high)

        # Calculate which bin each observation belongs to
        bin_indices = ((clipped_obs - self.low) // self.bin_widths).astype(int)

        # Ensure the indices are in valid ranges
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        return bin_indices

    def _convert_to_one_number(self, digits):
        # Convert a list of bin indices to a single integer
        return sum([d * (self.n_bins ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        # Discretize the observation and map it to a single integer
        bin_indices = self._discretize_observation(observation.flatten())

        return self._convert_to_one_number(bin_indices)
    
class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.action_space, Box), "The action space must be a Box space."

        low = self.action_space.low if low is None else np.array(low)
        high = self.action_space.high if high is None else np.array(high)

        # Clip values in low and high to be within the specified bounds
        max_abs_value = 1e6
        low = np.clip(low, -max_abs_value, max_abs_value)
        high = np.clip(high, -max_abs_value, max_abs_value)

        # Calculate the width of each bin
        self.bin_widths = (high - low) / n_bins

        self.n_bins = n_bins
        self.low = low
        self.high = high
        self.act_shape = np.prod(self.action_space.shape)

        # Define the new discrete action space
        self.action_space = Discrete(n_bins ** self.act_shape)
        print("New action space:", self.action_space)

    def _discretize_action(self, action):
        # Clip the action to ensure it stays within the adjusted bounds
        clipped_act = np.clip(action, self.low, self.high)

        # Calculate which bin each action belongs to
        bin_indices = ((clipped_act - self.low) // self.bin_widths).astype(int)

        # Ensure the indices are in valid ranges
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        return bin_indices

    def _convert_to_action(self, discrete_action):
        # Convert the single discrete action back into the continuous space
        indices = []
        for i in range(self.act_shape):
            indices.append(discrete_action % self.n_bins)
            discrete_action //= self.n_bins

        indices = np.array(indices)
        continuous_action = self.low + indices * self.bin_widths

        return continuous_action

    def action(self, action):
        # Convert the discrete action to a continuous one
        return self._convert_to_action(action)
    
class DiscretizedEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, n_bins_obs=10, n_bins_actions=10):
        super().__init__(env)

        self.env = env

        # Wrap the observation space if it's continuous
        if isinstance(env.observation_space, Box):
            self.env = DiscretizedObservationWrapper(self.env, n_bins=n_bins_obs)
            # Update observation space to the new discrete space
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = env.observation_space

        # Wrap the action space if it's continuous
        if isinstance(env.action_space, Box):
            self.env = DiscretizedActionWrapper(self.env, n_bins=n_bins_actions)
            # Update action space to the new discrete space
            self.action_space = self.env.action_space
        else:
            self.action_space = env.action_space

    def step(self, action):
        # Step in the wrapped environment
        return self.env.step(action)

    def reset(self, **kwargs):
        # Reset the environment
        return self.env.reset(**kwargs)
