from typing import Self, override

import numpy as np
from numba import njit


class person(object):
    """ """

    def __init__(self, rumour: str) -> None:
        """ """
        self._rumour: str = rumour
        self._has_heard_rumour: bool = False

    @override
    def __repr__(self) -> str:
        """ """
        return ""  # for now, this will suffice

    def has_rumour(self) -> bool:
        """ """
        return self._has_heard_rumour

    def converse(self, other: Self) -> None:
        """ """
        if other.has_rumour():
            self._has_heard_rumour = True
            self._rumour = other._rumour
        else:
            pass


MAX_CONTACTS: int = 11
MAX_ITERATIONS: int = 10_000
MAX_DAYS: int = 2_000


njit(fastmath=True)


def main() -> None:
    """ """
    np.array([])
    print("Hello!")


if __name__ == "__main__":
    main()
