from dataclasses import dataclass
import numpy as np
from typing import Sequence
from sgp4.api import Satrec, SatrecArray

@dataclass(frozen=True)
class TLEBatch:
    names: np.ndarray
    models: Sequence[Satrec]
    propagator: SatrecArray