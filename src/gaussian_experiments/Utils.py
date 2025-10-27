from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Any, Union
import os

PathLike = Union[str, Path]

def save_results_to_xlsx(records, output_dir, filename='results.xlsx'):
    """
    Save records to an Excel file built from an output directory and filename.
    """
    df = pd.DataFrame(records)

    def to_builtin(x):
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.integer):  return int(x)
        if isinstance(x, np.bool_):    return bool(x)
        return x

    df = df.applymap(to_builtin)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    df.to_excel(file_path, index=False)
    print(f"File saved to: {file_path}")
    return file_path

def save_pickle(obj: Any, output_dir: PathLike, filename: str = "object.pkl") -> Path:
    """
    Save a Python object to a pickle file built from an explicit output directory and filename.
    Creates parent directories if they don't exist.

    Parameters
    ----------
    obj : Any
        Python object to serialize.
    output_dir : str | Path
        Target directory where the pickle will be saved.
    filename : str
        Output filename (default: 'object.pkl').

    Returns
    -------
    Path
        Full path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    with file_path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return file_path


def load_pickle(output_dir: PathLike, filename: str = "object.pkl") -> Any:
    """
    Load a Python object from a pickle file using an explicit output directory and filename.

    Parameters
    ----------
    output_dir : str | Path
        Directory where the pickle is located.
    filename : str
        Filename to load (default: 'object.pkl').

    Returns
    -------
    Any
        Deserialized Python object.
    """
    file_path = Path(output_dir) / filename
    with file_path.open("rb") as f:
        return pickle.load(f)


def read_directories(directory, img=None, exclude_json=None):
    # Get a list of filenames in the specified directory
    filenames = []
    for filename in os.listdir(directory):
        if img is not None:
            # If 'img' is provided, filter filenames containing it
            if img in filename:   
                filenames.append(filename)
        elif exclude_json is not None:
            filenames.append(filename.replace('.json',''))     
        else:
            filenames.append(filename)    
    return filenames