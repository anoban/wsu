from typing import override


class person(object):
    def __init__(self, rumour: str) -> None:
        self._rumour: str = rumour
        self._has_heard_rumour: bool = False

    @override
    def __repr__(self) -> str:
        return ""  # for now, this will suffice

    def converse(self, other: person) -> None:
        pass


def main() -> None:
    pass


if __name__ == "__main__":
    main()
