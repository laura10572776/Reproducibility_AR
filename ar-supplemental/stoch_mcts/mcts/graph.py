from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import gym
import numpy as np


class Node:
    def __init__(
        self,
        env: gym.Env,
        state: Any,
        reward: float = 0.0,
        done: bool = False,
        prior: Optional[np.ndarray] = None,
    ):
        self.env_ = deepcopy(env)
        self.state = state
        self.reward = reward
        self.done = done
        self.prior = prior
        self.init_eval = None  # type: Optional[float]
        self.n_visits = 0
        self.n_action_visits = np.zeros(env.action_space.n)
        self.cum_action_vals = np.zeros(env.action_space.n)
        self.children = defaultdict(list)  # type: Dict[int, List[Node]]

    @property
    def env(self) -> gym.Env:
        return deepcopy(self.env_)

    def avg_vals(self) -> np.ndarray:
        return np.divide(
            self.cum_action_vals,
            self.n_action_visits,
            out=np.zeros_like(self.cum_action_vals),
            where=self.n_action_visits != 0,
        )

    def is_child(self, action: int, state: Any) -> bool:
        for node in self.children[action]:
            if state_equality(node.state, state):
                return True
        return False

    def get_child(self, action: int, state: Any) -> "Node":
        for node in self.children[action]:
            if state_equality(node.state, state):
                return node
        raise ValueError("Requested node does not exist")


def state_equality(x: Any, y: Any) -> bool:
    if isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray)
        return (x == y).all()
    if isinstance(x, dict):
        assert isinstance(y, dict)
        if set(x.keys()) != set(y.keys()):
            return False
        return all([state_equality(x[key], y[key]) for key in x.keys()])
    if isinstance(x, int):
        assert isinstance(y, int)
        return x == y
    if isinstance(x, str):
        assert isinstance(y, str)
        return x == y
    if isinstance(x, tuple):
        assert isinstance(y, tuple)
        if len(x) != len(y):
            return False
        return all([state_equality(x_i, y_i) for x_i, y_i in zip(x, y)])
    raise ValueError(f"Cannot handle input types {type(x)}")
