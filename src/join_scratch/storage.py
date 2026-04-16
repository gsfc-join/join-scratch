"""Storage abstraction for JOIN scratch code.

Provides a unified interface for reading data from local disk or S3.
File I/O is mediated through fsspec-compatible filesystem objects:
  - S3: obstore.fsspec.FsspecStore (obstore-backed, preferred)
  - local: fsspec built-in LocalFileSystem (fallback; no extra dependency)

Usage
-----
Build a StorageConfig from CLI args (or use defaults), then pass it to the
load_* functions in each regrid module::

    cfg = StorageConfig.from_args()
    files = cfg.glob("JOIN/AMSR2/**/*.h5")
    with cfg.open(files[0]) as f:
        ds = xr.open_dataset(f, engine="h5netcdf")

Supported storage types
-----------------------
local   Read from the local filesystem.  ``storage_location`` is treated as
        the root directory (default: ``_data-raw/`` relative to project root).

s3      Read from S3 via obstore.  ``storage_location`` is a bucket URL of
        the form ``s3://bucket-name/optional/prefix`` (default:
        ``s3://airborne-smce-prod-user-bucket/JOIN``).

source  Reserved for future direct-from-source access.  Raises
        NotImplementedError immediately.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default S3 location (matches scripts/download_test_data.sh)
# ---------------------------------------------------------------------------

S3_DEFAULT_BUCKET = "airborne-smce-prod-user-bucket"
S3_DEFAULT_PREFIX = "JOIN"
S3_DEFAULT_REGION = "us-west-2"

# Derived default location string for S3
S3_DEFAULT_LOCATION = f"s3://{S3_DEFAULT_BUCKET}/{S3_DEFAULT_PREFIX}"

StorageType = Literal["local", "s3", "source"]


# ---------------------------------------------------------------------------
# StorageConfig
# ---------------------------------------------------------------------------


@dataclass
class StorageConfig:
    """Resolved storage configuration.

    Attributes
    ----------
    storage_type:
        ``"local"``, ``"s3"``, or ``"source"``.
    storage_location:
        For ``local``: a :class:`pathlib.Path` to the root directory.
        For ``s3``: a string ``"s3://bucket/prefix"`` (no trailing slash).
        For ``source``: unused (always raises).
    """

    storage_type: StorageType
    storage_location: Path | str  # Path for local, str for s3/source

    # Lazily-constructed filesystem objects (not part of the public API)
    _fs: object = field(default=None, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def local(cls, root: Path | str | None = None) -> "StorageConfig":
        """Create a local storage config, defaulting to ``_data-raw/``."""
        if root is None:
            root = _project_root() / "_data-raw"
        return cls(storage_type="local", storage_location=Path(root))

    @classmethod
    def s3(cls, location: str | None = None) -> "StorageConfig":
        """Create an S3 storage config, defaulting to the JOIN bucket."""
        return cls(
            storage_type="s3",
            storage_location=location or S3_DEFAULT_LOCATION,
        )

    @classmethod
    def from_args(
        cls,
        storage_type: StorageType | None = None,
        storage_location: str | None = None,
    ) -> "StorageConfig":
        """Build a StorageConfig from the parsed CLI values.

        Parameters
        ----------
        storage_type:
            One of ``"local"``, ``"s3"``, ``"source"``, or ``None`` (defaults
            to ``"s3"``).
        storage_location:
            Override the default root / bucket+prefix.
        """
        stype: StorageType = storage_type or "s3"

        match stype:
            case "local":
                root = Path(storage_location) if storage_location else None
                return cls.local(root)
            case "s3":
                return cls.s3(storage_location)
            case "source":
                raise NotImplementedError(
                    "storage_type='source' (direct-from-source download) is not yet "
                    "implemented.  Use 'local' or 's3' instead."
                )
            case _:
                raise ValueError(
                    f"Unknown storage_type {stype!r}. "
                    "Choose from: 'local', 's3', 'source'."
                )

    # ------------------------------------------------------------------
    # Filesystem access
    # ------------------------------------------------------------------

    def _get_fs(self):
        """Return (and cache) the fsspec-compatible filesystem object."""
        if self._fs is not None:
            return self._fs

        if self.storage_type == "local":
            import fsspec

            self._fs = fsspec.filesystem("file")

        elif self.storage_type == "s3":
            from obstore.fsspec import FsspecStore

            self._fs = FsspecStore("s3", region=S3_DEFAULT_REGION)

        return self._fs

    def open(self, path: str | Path, mode: str = "rb") -> IO[bytes]:
        """Open a file and return a file-like object.

        Parameters
        ----------
        path:
            For ``local``: absolute path or path relative to
            ``storage_location``.  Strings are accepted for convenience.
            For ``s3``: a full ``s3://…`` URL, or a key relative to the
            configured bucket+prefix (e.g. ``"AMSR2/subdir/file.h5"`` or
            ``"lis_input_NMP_1000m_missouri.nc"``).
        mode:
            ``"rb"`` (default) or ``"r"``.
        """
        if self.storage_type == "source":
            raise NotImplementedError("storage_type='source' is not yet implemented.")

        fs = self._get_fs()

        if self.storage_type == "local":
            resolved = _resolve_local(Path(self.storage_location), path)
            log.debug("Opening local file: %s", resolved)
            return fs.open(str(resolved), mode)

        # S3 — build a full s3:// URL and pass it to the FsspecStore
        url = _resolve_s3_url(str(self.storage_location), str(path))
        log.debug("Opening S3 URL: %s", url)
        return fs.open(url, mode)

    def glob(self, pattern: str) -> list[str]:
        """Return a sorted list of paths / S3 keys matching *pattern*.

        Parameters
        ----------
        pattern:
            A glob pattern relative to ``storage_location``.
            For local, this is joined with the root directory.
            For S3, the pattern is matched against keys under the prefix.
            ``**`` is supported (recursive).
        """
        if self.storage_type == "source":
            raise NotImplementedError("storage_type='source' is not yet implemented.")

        if self.storage_type == "local":
            root = Path(self.storage_location)
            matches = sorted(str(p) for p in root.glob(pattern))
            log.debug("glob('%s') → %d local file(s)", pattern, len(matches))
            return matches

        # S3 — use obstore list + fnmatch
        from obstore.store import S3Store

        bucket, prefix = _parse_s3_location(str(self.storage_location))
        store = S3Store(bucket, region=S3_DEFAULT_REGION, prefix=prefix)

        # List all objects under the store prefix.
        # obstore returns keys *relative* to the store prefix (no leading prefix/).
        all_keys: list[str] = [obj["path"] for page in store.list(None) for obj in page]

        # The caller passes patterns like "JOIN/AMSR2/**/*.h5" where "JOIN" is
        # the store prefix already baked in.  Strip that prefix component from
        # the pattern so it matches the relative keys returned by obstore.
        rel_pattern = pattern
        if prefix:
            prefix_slash = prefix.rstrip("/") + "/"
            if rel_pattern.startswith(prefix_slash):
                rel_pattern = rel_pattern[len(prefix_slash) :]

        matched = sorted(k for k in all_keys if fnmatch.fnmatch(k, rel_pattern))
        log.debug(
            "S3 glob('%s') (rel: '%s') over prefix '%s' → %d key(s)",
            pattern,
            rel_pattern,
            prefix,
            len(matched),
        )
        # Return full s3:// URLs so callers can pass them straight to open()
        base_url = f"s3://{bucket}/{prefix}".rstrip("/")
        return [f"{base_url}/{k}" for k in matched]


# ---------------------------------------------------------------------------
# CLI argument helpers
# ---------------------------------------------------------------------------


def add_storage_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--storage-type`` and ``--storage-location`` to *parser*."""
    parser.add_argument(
        "--storage-type",
        choices=["local", "s3", "source"],
        default="s3",
        help=(
            "Where to read input data from.  "
            "'local': local disk (see --storage-location).  "
            "'s3': AWS S3 (default; uses the JOIN project bucket).  "
            "'source': reserved for future direct download (not yet implemented)."
        ),
    )
    parser.add_argument(
        "--storage-location",
        default=None,
        help=(
            "Override the default storage root.  "
            "For 'local': path to the data-raw directory (default: _data-raw/).  "
            f"For 's3': bucket+prefix URL (default: {S3_DEFAULT_LOCATION})."
        ),
    )


def storage_config_from_namespace(ns: argparse.Namespace) -> StorageConfig:
    """Build a :class:`StorageConfig` from a parsed :class:`argparse.Namespace`."""
    return StorageConfig.from_args(
        storage_type=ns.storage_type,
        storage_location=ns.storage_location,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the repository root (two levels above this file's package)."""
    return Path(__file__).resolve().parents[2]


def _parse_s3_location(location: str) -> tuple[str, str]:
    """Parse ``"s3://bucket/prefix"`` into ``(bucket, prefix)``.

    Returns ``(bucket, "")`` if there is no prefix component.
    """
    stripped = location.removeprefix("s3://")
    parts = stripped.split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    return bucket, prefix


def _resolve_local(root: Path, path: str | Path) -> Path:
    """Resolve a path relative to *root* unless it is already absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return root / p


def _resolve_s3_url(location: str, path: str) -> str:
    """Given a storage_location and a user-supplied path, return a full s3:// URL.

    If *path* is already a full ``s3://…`` URL it is returned unchanged.
    Otherwise it is appended to *location* (which must be an ``s3://…`` URL).
    """
    if path.startswith("s3://"):
        return path
    # Strip leading slash to avoid double-slash
    return location.rstrip("/") + "/" + path.lstrip("/")
