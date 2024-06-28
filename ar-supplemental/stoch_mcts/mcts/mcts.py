import gym
import numpy as np

from .backpropagator import Backpropagator
from .evaluator import Evaluator
from .graph import Node
from .path import Path
from .selector import Selector

import time

class MCTS:
    def __init__(
        self,
        selector: Selector,
        evaluator: Evaluator,
        backpropagator: Backpropagator,
    ):
        self.selector = selector
        self.evaluator = evaluator
        self.backpropagator = backpropagator

    def __call__(self, env: gym.Env, state: np.ndarray, n_rollouts: int) -> Node:
        root = Node(env, state)
        for t in range(n_rollouts):
            self.rollout(root)
        return root

    def rollout(self, root: Node) -> None:
        path = self.selector(root)
        evaluation = self.evaluator(path.tail)
        self.backpropagator(path, evaluation)
