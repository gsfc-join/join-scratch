"""Dataset handlers for join_scratch instruments."""

from join_scratch.datasets.amsr2 import Amsr2FileHandler
from join_scratch.datasets.ceda import CedaFileHandler
from join_scratch.datasets.icesat2 import Icesat2FileHandler
from join_scratch.datasets.viirs import ViirsFileHandler

__all__ = [
    "Amsr2FileHandler",
    "CedaFileHandler",
    "ViirsFileHandler",
    "Icesat2FileHandler",
]
