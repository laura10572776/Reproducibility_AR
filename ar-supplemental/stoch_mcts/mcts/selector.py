from typing import Any, Callable, Optional, Tuple

import gym
import numpy as np

from .graph import Node
from .path import Path

ActionSelector = Callable[[Node], int]

def dealer_policy(node: Node) -> int:
    if node.state[0] < 17:
        return 1
    else:
        return 0

def wikipedia_policy(node: Node) -> int:
    if node.state[1] < 3.5:
        return int(node.state[0] < 12.5)
    elif node.state[1] < 6.5:
        return int(node.state[0] < 11.5)
    return int(node.state[0] < 16.5)

class UCT:
    def __init__(self, c: float):
        self.c = c

    def __call__(self, node: Node) -> int:
        avg_vals = node.avg_vals()
        n_visits = node.n_visits
        n_pulls = node.n_action_visits
        if 0 in n_pulls:
            return np.random.choice(np.flatnonzero(0 == n_pulls))
        priority = avg_vals + self.c * np.sqrt(np.log(n_visits) / n_pulls)
        return np.random.choice(np.flatnonzero(priority == priority.max()))

def hitter(node: Node):
    return 1


StateSelector = Callable[[Node, int], Tuple[Node, bool]]


def sampler(node: Node, action: int) -> Tuple[Node, bool]:
    env = node.env
    state, reward, done, info = env.step(action)
    if node.is_child(action, state):
        return node.get_child(action, state), False
    else:
        child = Node(env, state, reward, done)
        node.children[action].append(child)
        return child, True


class ProgressiveWidening:
    def __init__(self, k: float, alpha: float):
        self.k = k
        self.alpha = alpha

    def __call__(self, node: Node, action: int) -> Tuple[Node, bool]:
        if (
            len(node.children[action])
            <= self.k * node.n_action_visits[action] ** self.alpha
        ):
            return sampler(node, action)
        return np.random.choice(node.children[action]), False


class AbstractionRefining:
    def __init__(self, k: float, alpha: float, dist: Optional[Callable[[Any, Any], float]]=None):
        self.k = k
        self.alpha = alpha
        if dist is None:
            self.dist = np.linalg.norm
        else:
            self.dist = dist

    def __call__(self, node: Node, action: int) -> Tuple[Node, bool]:
        if len([node_ for node_ in node.children[action] if not node_.done]) == 0:
            return sampler(node, action)
        env = node.env
        state, reward, done, info = env.step(action)
        if not done:
            distances = {
                node_: self.dist(state, node_.state) for node_ in node.children[action] if not node_.done
            }
            nn = min(distances, key=distances.get)
            if min(distances.values()) <= self.k * nn.n_visits ** (-self.alpha):
                return nn, False
        child = Node(env, state, reward, done)
        node.children[action].append(child)
        return child, True


class Selector:
    def __init__(self, action_selector: ActionSelector, state_selector: StateSelector):
        self.state_selector = state_selector
        self.action_selector = action_selector

    def __call__(self, root: Node) -> Path:
        node = root
        path = Path(node)
        while True:
            action = self.action_selector(node)
            node_, new_node = self.state_selector(node, action)
            path.append(node, action, node_)
            node = node_
            if node.done or new_node:
                break
        return path
