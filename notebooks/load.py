import sys

def append_lib_path() -> None:
    """ 
    append the path to the lib directory to PATH env variable,
    so the modules in package lib could be loaded as needed.
    this is a helper function, in an isolated script.
    """
    sys.path.append(r"../lib/")
