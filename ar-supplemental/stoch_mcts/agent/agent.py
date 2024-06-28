import gym
import numpy as np

from ..mcts.mcts import MCTS

class Agent:
    def __init__(self, mcts: MCTS):
        self.mcts = mcts

    def act(self, env: gym.Env, state: np.ndarray, n_rollouts: int) -> int:
        root = self.mcts(env, state, n_rollouts)
        vals = root.avg_vals()
        return np.random.choice(np.flatnonzero(vals == vals.max()))
