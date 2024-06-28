from typing import List

import numpy as np

from .graph import Node


class Path:
    def __init__(self, head: Node):
        self.nodes = []  # type: List[Node]
        self.actions = []  # type: List[int]
        self.tail = head

    def append(self, node: Node, action: int, node_: Node):
        self.nodes.append(node)
        self.actions.append(action)
        self.tail = node_
