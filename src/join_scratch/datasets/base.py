"""Base class for join_scratch file handlers."""

import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class JoinFileHandler(ABC):
    """Abstract base class for join_scratch file handlers.

    A lightweight ABC that does not require satpy YAML infrastructure.
    """

    def __init__(self, filename, filename_info=None, filetype_info=None):
        self.filename = filename
        self.filename_info = filename_info if filename_info is not None else {}
        self.filetype_info = filetype_info if filetype_info is not None else {}

    @classmethod
    def from_path(cls, path, fs=None):
        """Create a handler from a local path or an fsspec filesystem path."""
        if fs is None:
            return cls(
                filename=str(path),
                filename_info={"filename": str(path)},
                filetype_info={},
            )
        else:
            from satpy.readers.core.remote import FSFile

            fsfile = FSFile(str(path), fs=fs)
            return cls(
                filename=fsfile,
                filename_info={"filename": str(path)},
                filetype_info={},
            )

    @property
    def start_time(self):
        return self.filename_info.get("start_time")

    @property
    def end_time(self):
        return self.filename_info.get("end_time")

    @abstractmethod
    def get_dataset(self, dataset_id, ds_info=None):
        """Load and return the dataset identified by dataset_id."""

    @abstractmethod
    def get_area_def(self, dataset_id=None):
        """Return the pyresample AreaDefinition for this file."""
