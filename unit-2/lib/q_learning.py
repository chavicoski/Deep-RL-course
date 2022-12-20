"""Module that implements the Q-Learning algorithm for training RL Agents"""
import random
from typing import List, Tuple

import imageio
import numpy as np
from gym import Env
from pyvirtualdisplay import Display
from tqdm import tqdm


class QLearning:
    def __init__(self, env: Env) -> None:
        """Q-Learning agent constructor"""
        self.env = env
        self._initialize_q_table()

    def _initialize_q_table(self) -> None:
        """Initialize the Agent Q table with a 0-matrix of shape (n_states, n_actions)"""
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def _greedy_policy(self, state: int) -> int:
        """Greedy policy to select the most valuable action for the provided `state`

        Args:
            state (int): Integer representation of the state to select the best action

        Returns:
            The most valuable action for the `state` following a greedy policy
        """
        return np.argmax(self.q_table[state])

    def _epsilon_greedy_policy(self, state: int, epsilon: float) -> int:
        """Policy that deals with the exploration/exploitation trade-of by choosing an
        action based on a probability `epsilon`:
           if 1 - epsilon:
               # Exploitation based action
               select a greedy action
           else:
               # Exploration based action
               select a random action

        Args:
            state (int): Integer representation of the state to select the action
            epsilon (float): Probability (in range [0, 1]) to select a random action

        Returns:
            The selected action
        """
        if random.uniform(0, 1) > epsilon:
            return self._greedy_policy(state)
        else:
            return self.env.action_space.sample()

    def train(
        self,
        n_episodes: int,
        max_steps: int,
        learning_rate: float,
        gamma: float,
        min_epsilon: float,
        max_epsilon: float,
        decay_rate: float,
    ) -> None:
        """Train algorithm to estimate the values of the Q-table

        Args:
            n_episodes (int): Number of episodes to train
            max_steps (int): Max steps per episode
            learning_rate (float): Parameter to control the convergence speed
            gamma (float): Discounting rate for future return
            min_epsilon (float): Minimum value of `epsilon`
            max_epsilon (float): Maximum value of `epsilon` (starting value)
            decay_rate (float): Exponential decay rate value for `epsilon`
        """
        delta_epsilon = max_epsilon - min_epsilon
        for episode in tqdm(range(n_episodes)):
            # Exponentially reduce epsilon (more episodes => less exploration)
            epsilon = min_epsilon + delta_epsilon * np.exp(-decay_rate * episode)
            # Reset the environment for the new episode
            state = self.env.reset()

            for _ in range(max_steps):
                # Choose the next action
                action = self._epsilon_greedy_policy(state, epsilon)

                # Take the action and recieve the feedback from the environment
                new_state, reward, done, _ = self.env.step(action)

                # Update the Q-table following the equation:
                # Q(s,a) := Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.q_table[state][action] += learning_rate * (
                    reward  # R(s,a)
                    + gamma * np.max(self.q_table[new_state])  # gamma * max Q(s',a')
                    - self.q_table[state][action]  # Q(s,a)
                )

                # If done, finish the episode
                if done:
                    break

                # Update the current state for the next step
                state = new_state

    def _play_episode(
        self,
        max_steps: int,
        record_video: bool = False,
        seed: int = None,
    ) -> dict:
        """Execute one episode run

        Args:
            max_steps (int): Max steps for the episode
            record_video (bool): To return the video of the episode
            seed (int): Seed to initialize the environment

        Returns:
            A dictionary with:
                "reward": Total reward of the episode
                "steps": Number of executed steps
                "video": Video of the episode run (If enabled)
        """
        state = self.env.reset(seed=seed)
        total_reward = 0  # Accumulate the episode rewards
        frames = []  # To record the video
        for step in range(max_steps):
            # Choose the next action that maximizes the expected future reward
            action = self._greedy_policy(state)
            # Take the action and recieve the feedback from the environment
            new_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if record_video:
                frames.append(self.env.render(mode="rgb_array"))

            # If done, finish the episode
            if done:
                break

            # Update the current state for the next step
            state = new_state

        # Collect the frames into a numpy array (if `record_video` is enabled)
        video = np.array(frames) if record_video else None

        return {"reward": total_reward, "steps": step + 1, "video": video}

    def evaluate(
        self,
        n_episodes: int,
        max_steps: int,
        seeds: List[int] = None,
    ) -> Tuple[float, float]:
        """Evaluate the Q-Learning agent for `n_episodes` episodes and return the
        average reward and std of the reward

        Args:
            n_episodes (int): Number of episodes to train
            max_steps (int): Max steps per episode
            seeds (List[int]): List of length `n_episodes` with the seeds to initialize
                               each episode

        Returns:
            The mean reward and std values
        """
        if seeds and len(seeds) != n_episodes:
            raise ValueError(
                f"The number of seeds ({len(seeds)}) should be equal to the number "
                f"of episodes ({n_episodes})"
            )

        episode_rewards = []
        for episode in tqdm(range(n_episodes)):
            # Play the episode and get the results
            episode_res = self._play_episode(
                max_steps,
                record_video=False,
                seed=seeds[episode] if seeds else None,
            )
            # Store the episode total reward
            episode_rewards.append(episode_res["reward"])

        # Compute the average results
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward

    def record_video(self, out_path: str, max_steps: int, fps: int = 1) -> None:
        """Plays one episode and saves the resulting video in a file

        Args:
            out_path (str): Filepath to store the video (with .mp4 extension)
            max_steps (int): Max steps for the episode
            fps (int): Frames per second for the video
        """
        # Prepare the virtual display to render the video
        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()
        # Play one episode and get the video frames
        episode_res = self._play_episode(max_steps, record_video=True)
        # Create the video file
        imageio.mimsave(out_path, episode_res["video"], fps=fps)
