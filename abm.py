from random import randint
from typing import Self, override

import numpy as np
from numba import njit
from numpy.typing import NDArray


class person(object):
    """ """

    @override
    def __init__(self, rumour: str, has_rumour: int = False) -> None:
        """ """
        self._rumour: str = rumour
        self._has_heard_rumour: bool = False

    @override
    def __repr__(self) -> str:
        """ """
        return f"a person with{'' if self._has_heard_rumour else 'out'} a rumour"

    def has_rumour(self) -> bool:
        """ """
        return self._has_heard_rumour

    # other alternatives to typing.Self is using a string literal "person" which will be eval()ed at runtime
    # or import annotations from __future__ which turns all type annotations into string literals
    def converse(self, other: Self) -> None:
        """ """
        if other._has_heard_rumour:
            self._has_heard_rumour = True
            self._rumour = other._rumour
        else:
            pass

    def rumour(self) -> str:
        """ """
        return self._rumour


MAX_POPULATION_SIZE: int = 8_000_000
MAX_ITERATIONS: int = 10_000
MAX_CONTACTS: int = 10
MAX_DAYS: int = 2_000


njit(fastmath=True, parallel=True, nogil=True)


def main() -> None:
    """ """
    population = np.empty(shape=(MAX_POPULATION_SIZE), dtype=person)
    population[randint(a=0, b=MAX_POPULATION_SIZE)] = person("There's a snake in the grass!")
    daily_counts: NDArray[np.int64] = np.zeros(shape=MAX_DAYS, dtype=np.int64)

    for d in range(MAX_DAYS):  # for each day
        for _ in range(MAX_ITERATIONS):  # for every iteration
            # we randomly choose a MAX_CONTACTS number of people for conversations
            chosen: person = population[randint(a=0, b=MAX_POPULATION_SIZE)]
            for contact in population[np.random.randint(low=0, high=MAX_POPULATION_SIZE, size=MAX_CONTACTS)]:
                chosen.converse(contact)
        # update the count of individuals who have heard the rumour by now
        daily_counts[d] = population.sum()


if __name__ == "__main__":
    main()
