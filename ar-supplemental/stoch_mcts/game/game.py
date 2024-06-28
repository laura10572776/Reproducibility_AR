import random
from typing import Any, Tuple

from gym import Env, spaces
import numpy as np

class ContinuousBlackJack(Env):
    metadata = {"render.modes": ["human"]}
    STAND = 0
    HIT = 1
    START_VALUE = 0.0
    LOSE_VALUE = -1
    WIN_VALUE = 1
    MIN_DEAL = 1.0
    MAX_DEAL = 11.0
    MAX_VALID = 21.0
    DEALER_STOP = 17.0
    NUM_ACTIONS = 2

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=self.START_VALUE,
            high=self.MAX_DEAL + self.MAX_VALID,
            shape=(2,),
            dtype=np.float32,
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if len(self.history) == 0:
            deal = self.deal_()
            self.dealer_total += deal
            self.history += f"dealer: {deal} ({self.dealer_total});\n"
            return (np.array([self.player_total, self.dealer_total]), 0, False, {})
        if action == self.STAND:
            while self.dealer_total < self.DEALER_STOP:
                deal = self.deal_()
                self.dealer_total += deal
                self.history += f"dealer: {deal} ({self.dealer_total});\n"
            dealer_won = self.player_total < self.dealer_total <= self.MAX_VALID
            reward = self.LOSE_VALUE if dealer_won else self.WIN_VALUE
            return (np.array([self.player_total, self.dealer_total]), reward, True, {})
        elif action == self.HIT:
            deal = self.deal_()
            self.player_total += deal
            self.history += f"player: {deal} ({self.player_total});\n"
            done = self.player_total > self.MAX_VALID
            reward = self.LOSE_VALUE if done else 0
            return (np.array([self.player_total, 0.0]), reward, done, {})
        raise ValueError("Received invalid action")

    def reset(self) -> np.ndarray:
        self.player_total = self.START_VALUE
        self.dealer_total = self.START_VALUE
        self.history = ""
        return np.array([self.player_total, 0.0])

    def render(self) -> None:
        print(self.history, end="")

    def close(self) -> None:
        pass

    def deal_(self) -> float:
        return random.uniform(self.MIN_DEAL, self.MAX_DEAL)

    def distance(self, x, y):
        return np.abs(x - y).sum()


class Hallway(Env):
    metadata = {"render.modes": ["human"]}
    START_POS = 0
    LEFT = 0
    RIGHT = 1

    def __init__(self, length=2, penalty=1, noise=0.01):
        super().__init__()
        self.length = length
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-self.length,
            high=self.length,
            shape=(1,),
            dtype=np.float32,
        )
        self.pos = self.START_POS
        self.noise = noise
        self.penalty = penalty

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if action == self.RIGHT:
            self.pos += 1 + np.random.normal(scale=self.noise)
        if action == self.LEFT:
            self.pos -= 1 + np.random.normal(scale=self.noise)
        else:
            raise ValueError("Received invalid action")
        finished = self.pos >= self.length
        reward = int(finished) - self.penalty
        return np.array([self.pos]), reward, finished, {}

    def reset(self) -> np.ndarray:
        self.pos = self.START_POS
        self.history = ""
        return np.array([self.pos])

    def render(self) -> None:
        print(self.history, end="")

    def close(self) -> None:
        pass

    def deal_(self) -> float:
        return random.uniform(self.MIN_DEAL, self.MAX_DEAL)


class Trap(Env):
    metadata = {"render.modes": ["human"]}
    START_POS = 0
    START_TIME = 0

    def __init__(
        self,
        n_actions=5,
        high_reward=100,
        avg_reward=70,
        ramp_length=1,
        trap_width=0.7,
        noise_amplitude=0.01,
    ):
        super().__init__()
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=self.START_POS,
            high=self.START_POS + 2 + 2 * noise_amplitude,
            shape=(2,),
            dtype=np.float32,
        )
        self.n_actions = n_actions
        self.high_reward = high_reward
        self.avg_reward = avg_reward
        self.ramp_length = ramp_length
        self.trap_width = trap_width
        self.noise_amplitude = noise_amplitude

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.pos[0] += action / (
            self.n_actions - 1
        ) + self.noise_amplitude * random.uniform(-1, 1)
        done = self.time == 1
        self.time += 1
        if self.pos[0] <= self.ramp_length:
            self.history += f"{round(self.pos[0], 2)} (ramp) "
            reward = self.avg_reward
            self.pos[1] = reward
        elif self.pos[0] <= self.ramp_length + self.trap_width:
            self.history += f"{round(self.pos[0], 2)} (trap) "
            reward = 0
            self.pos[0] = reward
        else:
            self.history += f"{round(self.pos[0], 2)} (goal) "
            reward = self.high_reward
            self.pos[1] = reward
        return np.array(self.pos), reward, done, {}

    def reset(self) -> np.ndarray:
        self.pos = [self.START_POS, self.avg_reward]
        self.time = self.START_TIME
        self.history = ""
        return np.array([self.pos])

    def render(self) -> None:
        print(self.history, end="")

    def close(self) -> None:
        pass

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y)
        
        
