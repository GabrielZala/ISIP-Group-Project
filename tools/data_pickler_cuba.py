import pickle
from pathlib import Path


def save(object_to_pickle, filename, subfolder='tools/pickle'):

    # make path
    root = Path(".")
    path_pickle = root / subfolder / filename

    # Write or overwrite new file
    file_new = open(path_pickle, "wb")
    pickle.dump(object_to_pickle, file_new)
    file_new.close()


def load(filename, subfolder='tools/pickle'):

    # make path
    root = Path(".")
    path_pickle = root / subfolder / filename

    # Write or overwrite new file
    file_to_load = open(path_pickle, "rb")
    file = pickle.load(file_to_load)
    file_to_load.close()

    return file


