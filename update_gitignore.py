# run this script to append large files to the .gitignore in the current directory
# this will not overwrite previous contents of the .gitignore file but will not add a file if it already exists
# and will remove files that no longer exist on disk


def __get_gitignore_contents(fpath: str, ignore_wildcards: bool = True, ignore_comments: bool = True) -> list[str]:
    """
    :param path: path to .gitignore file
    :type path: str
    :param ignore_wildcards: whether to ignore wild cards e.g. *.txt
    :type ignore_wildcards: bool
    :param ignore_comments: whether to ignore comments
    :type ignore_comments: bool
    :return: lines in the .gitignore
    :rtype: list[str]
    """

    with open(file=fpath, mode="r") as fp:
        contents: list[str] = fp.read().splitlines()  # read in the contents and split the lines


def main() -> None:
    """
    Docstring for main
    """

    __get_gitignore_contents()

    pass


if __name__ == r"__main__":
    main()
