from typing import List

import gym
import numpy as np
from tqdm import tqdm

from ..agent.agent import Agent

import time

def run(env: gym.Env, agent: Agent, num_episodes: int, n_rollouts: int) -> List[float]:
    cum_rewards = []
    for i_episode in tqdm(range(num_episodes)):
        cum_reward = 0
        state = env.reset()
        while True:
            action = agent.act(env, state, n_rollouts)
            state, reward, done, info = env.step(action)
            cum_reward += reward
            if done:
                break
        cum_rewards.append(cum_reward)
    env.close()
    return cum_rewards