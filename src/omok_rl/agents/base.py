from abc import ABC, abstractmethod

from omok.env import BoardGame


class Agent(ABC):
    """A player. Given the current game, it returns the position (0 <= pos < size*size) to play.

    Agents get the whole env so they can read the board, the legal mask, or clone it for search.
    They must not modify it.
    """

    name = 'agent'

    @abstractmethod
    def act(self, env: BoardGame) -> int: ...

    def __repr__(self):
        return self.name
