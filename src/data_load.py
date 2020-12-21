import glob
import os
from typing import List

from src.config import RESOURCES_DIR
import numpy as np
from skimage import io


def load_all() -> List[np.ndarray]:
    pattern = os.path.join(RESOURCES_DIR, "*.png")
    filenames = glob.glob(pattern)
    return [io.imread(filename, as_gray=True) for filename in filenames]


def load_single(case_number: int) -> np.ndarray:
    filename = os.path.join(RESOURCES_DIR, f"{str(case_number)}.png")
    return io.imread(filename, as_gray=True)


