"""Implementation of the Reinforce algorithm to create, train and evaluate a Policy"""
from collections import deque
from typing import List, Tuple

import gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        h_size: int,
        n_layers: int = 2,
    ) -> None:
        super(Policy, self).__init__()
        if n_layers < 2:
            raise ValueError(f"`n_layers` must be at least 2! ({n_layers=})")

        # Add the input layer
        layers = [nn.Linear(state_size, h_size), nn.ReLU()]
        # Add hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(h_size, h_size))
            layers.append(nn.ReLU())
        # Add output layer
        layers.append(nn.Linear(h_size, action_size))
        layers.append(nn.Softmax(dim=1))

        # Create the model from the sequence of layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def act(self, state: np.ndarray, device: str) -> Tuple[int, float]:
        # Prepare the state as a torch tensor with the batch dimension
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Compute the policy output probabilities
        probs = self.forward(state).cpu()
        # Sample an action from the probability distribution
        m = Categorical(probs)
        action = m.sample()
        # Return the selected action and its corresponding probability
        return action.item(), m.log_prob(action)


def reinforce(
    env: gym.Env,
    policy: Policy,
    optimizer: optim.Optimizer,
    n_training_episodes: int,
    max_steps: int,
    gamma: float,
    print_every: int,
    device: str,
) -> List[float]:
    """Implementation of the Reinforce algorithm to train an agent

    Args:
        env (gym.Env): Training environment
        policy (Policy): Model to use as the policy to train
        optimizer (optim.Optimizer): Parameters optimizer
        n_training_episodes (int): Total number of training episodes
        max_steps (int): Maximum number of steps per episode
        gamma (float): Reward discount factor
        print_every (int): Number of episodes to run before printing metrics
        device (str): Computing device id

    Returns:
        The list of rewards for each training episode
    """
    scores = []  # To store all the episodes scores
    scores_deque = deque(maxlen=100)  # Last N scores to show mean reward progress
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []  # Stores the log_prob of each action taken
        rewards = []  # Stores the reward of each action taken
        state = env.reset()
        # Play an episode and collect the data
        for t in range(max_steps):
            # Choose an action
            action, log_prob = policy.act(state, device)
            saved_log_probs.append(log_prob)
            # Apply the action and get the results
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        episode_total_reward = sum(rewards)
        scores_deque.append(episode_total_reward)
        scores.append(episode_total_reward)

        # Calculate the return
        returns = deque(maxlen=max_steps)
        n_steps = len(rewards)
        """
        Compute the discounted returns at each timestep as:

            The sum of the gamma-discounted return at time t (G_t) + the reward at time t
        
        In O(N) time, where N is the number of time steps (this definition of the
        discounted return G_t follows the definition of this quantity shown at page 44
        of Sutton&Barto 2017 2nd draft)
        G_t = r_(t+1) + r_(t+2) + ...

        Given this formulation, the returns at each timestep t can be computed
        by re-using the computed future returns G_(t+1) to compute the current return G_t
        G_t = r_(t+1) + gamma*G_(t+1)
        G_(t-1) = r_t + gamma* G_t
        (this follows a dynamic programming approach, with which we memorize solutions in order
        to avoid computing them multiple times)

        This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...

        Given the above, we calculate the returns at timestep t as:
            
            gamma[t] * return[t] + reward[t]
        
        We compute this starting from the last timestep to the first, in order to employ
        the formula presented above and avoid redundant computations that would be needed
        if we were to do it from first to last.

        Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        a normal python list would instead require O(N) to do this.
        """
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        # Standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()  # Smallest representable float
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Compute the loss
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Apply Gradient Descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}")

    return scores


def evaluate_agent(
    env: gym.Env,
    policy: Policy,
    n_eval_episodes: int,
    max_steps: int,
    device: str,
) -> Tuple[float, float]:
    """Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward
    and std of reward

    Args:
        env (gym.Env): Evaluation environment
        policy (Policy): Policy to evaluate
        n_eval_episodes (int): Number of episodes to evaluate
        max_steps (int): Maximum number of steps per episode
        device (str): Computing device id

    Returns:
        Mean reward and std values
    """
    episode_rewards = []
    for _ in range(n_eval_episodes):
        state = env.reset()
        total_rewards_ep = 0

        for _ in range(max_steps):
            action, _ = policy.act(state, device)
            new_state, reward, done, _ = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def record_video(
    env: gym.Env,
    policy: Policy,
    out_path: str,
    fps: int = 30,
    device: str = "cpu",
) -> None:
    """Plays one episode and saves the resulting video in a file

    Args:
        env (gym.Env): Environment to play
        policy (Policy): Policy algorithm to take the actions for the video
        out_path (str): Filepath to store the video (with .mp4 extension)
        fps (int): Frames per second for the video
        device (str): Computing device id
    """
    print("Going to record a replay video...")
    state = env.reset()
    images = []  # To store the video frames
    # Store the starting frame before taking any action
    images.append(env.render(mode="rgb_array"))
    done = False
    while not done:
        # Take the action that have the maximum expected future reward given that state
        action, _ = policy.act(state, device)
        state, _, done, _ = env.step(action)
        # Store the frame of the current state
        images.append(env.render(mode="rgb_array"))

    # Store the frames in a video file
    imageio.mimsave(out_path, [np.array(img) for img in images], fps=fps)
    print(f"Video stored in '{out_path}'")
