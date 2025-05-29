from dataclasses import dataclass, field
from typing import Optional
import pickle
import numpy as np
import os

def load_calibration_matrices(calib_dir):
    intrinsics = {}
    extrinsics = {}

    for filename in os.listdir(calib_dir):
        if not filename.endswith('.pkl'):
            continue

        filepath = os.path.join(calib_dir, filename)
        if '_intrinsics.pkl' in filename:
            view = filename.replace('_intrinsics.pkl', '')
            with open(filepath, 'rb') as f:
                mat = pickle.load(f)['matrix']
                intrinsics[view] = mat.tolist() if isinstance(mat, np.ndarray) else mat
        elif '_extrinsics.pkl' in filename:
            view = filename.replace('_extrinsics.pkl', '')
            with open(filepath, 'rb') as f:
                mat = pickle.load(f)
                extrinsics[view] = mat.tolist() if isinstance(mat, np.ndarray) else mat

    return intrinsics, extrinsics

@dataclass
class CameraConfig:
    name: str
    type: str  # "kinova", "realsense", "logitech"
    serial: Optional[str] = None

@dataclass
class RealEnvConfig:
    wbc: int
    cameras: list[CameraConfig]
    pcl_cameras: list[str]
    data_folder: str
    calib_dir: Optional[str] = None
    is_sim: int = 0
    min_bound: list[float] = field(default_factory=list)
    max_bound: list[float] = field(default_factory=list)

    # These will be loaded dynamically and excluded from serialization
    intrinsics: dict = field(init=False, default_factory=dict, repr=False, compare=False)
    extrinsics: dict = field(init=False, default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.calib_dir is not None:
            self.intrinsics, self.extrinsics = load_calibration_matrices(self.calib_dir)

    def to_serializable_dict(self):
        d = asdict(self)
        # Manually remove non-serializable dynamic fields
        d.pop("intrinsics", None)
        d.pop("extrinsics", None)
        return d
