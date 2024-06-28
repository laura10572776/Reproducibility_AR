from typing import Callable

from .path import Path

Backpropagator = Callable[[Path, float], None]


class TreeBackpropagator:
    def __init__(self, discount: float):
        self.discount = discount

    def __call__(self, path: Path, evaluation: float) -> None:
        path.tail.init_eval = evaluation
        running_eval = path.tail.reward + self.discount * evaluation
        path.tail.n_visits += 1
        for node, action in reversed(tuple(zip(path.nodes, path.actions))):
            node.n_visits += 1
            node.n_action_visits[action] += 1
            node.cum_action_vals[action] += running_eval
            running_eval = node.reward + self.discount * running_eval
