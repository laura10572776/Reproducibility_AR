import random
from typing import Callable

import numpy as np
import gym

from .graph import Node

Evaluator = Callable[[Node], float]

import time

def null(node: Node) -> float:
    return 0

class RandomRollouts:
    def __init__(self, discount: float, n_simulations: float = 1):
        self.discount = discount
        self.n_simulations = n_simulations

    def __call__(self, node: Node) -> float:
        if node.done:
            return 0
        vals = []
        for _ in range(self.n_simulations):
            t = 0
            env = node.env
            cumulative_reward = 0
            done = False
            while not done:
                _, reward, done, _ = env.step(env.action_space.sample())
                cumulative_reward += (self.discount ** t) * reward
                t += 1
            vals.append(cumulative_reward)
        return np.mean(vals)


class Stayer:
    def __init__(self, discount: float):
        self.discount = discount

    def __call__(self, node: Node) -> float:
        if node.done:
            return 0
        env = node.env
        cumulative_reward = 0
        state = node.state
        done = False
        t = 0
        while not done:
            action = 0
            state, reward, done, _ = env.step(action)
            cumulative_reward += (self.discount ** t) * reward
            t += 1
        return cumulative_reward

class DealerPolicy:
    def __init__(self, discount: float):
        self.discount = discount

    def __call__(self, node: Node) -> float:
        if node.done:
            return 0
        env = node.env
        cumulative_reward = 0
        state = node.state
        done = False
        t = 0
        while not done:
            if sum(state) > 13:
                action = 0
            elif sum(state) < 10:
                return 1
            else:
                action = np.random.choice([0, 1])
            state, reward, done, _ = env.step(action)
            cumulative_reward += (self.discount ** t) * reward
            t += 1
        return cumulative_reward

def blackjack_evaluator(node: Node) -> float:
    if node.done:
        return 0
    state = node.state
    if state[0] < 17:
        return -0.2
    return (1.2 / 4) * (state[0] - 17) - 0.2
    