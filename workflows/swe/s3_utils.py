"""Shared S3 utilities for SWE workflow scripts using obstore."""

from __future__ import annotations

import logging
from typing import Generator

log = logging.getLogger(__name__)

# Default S3 region for the SMCE bucket
_DEFAULT_REGION = "us-west-2"


def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def make_store(bucket: str, prefix: str = "", region: str = _DEFAULT_REGION):
    """Create an obstore S3Store for a bucket/prefix.

    Parameters
    ----------
    bucket:
        S3 bucket name (without ``s3://`` prefix).
    prefix:
        Optional key prefix within the bucket (e.g. ``"JOIN"``).
    region:
        AWS region string.

    Returns
    -------
    obstore.store.S3Store
    """
    from obstore.store import S3Store

    return S3Store(bucket, region=region, prefix=prefix)


def make_fs(region: str = _DEFAULT_REGION):
    """Return an obstore FsspecStore for use with xarray / h5py / satpy.

    Parameters
    ----------
    region:
        AWS region string.

    Returns
    -------
    obstore.fsspec.FsspecStore
    """
    from obstore.fsspec import FsspecStore

    return FsspecStore("s3", region=region)


def list_s3(store, prefix: str = "") -> list[str]:
    """List all object keys under *prefix* in *store*.

    Parameters
    ----------
    store:
        An ``obstore.store.S3Store`` instance.
    prefix:
        Key prefix to list under (relative to the store's own prefix).

    Returns
    -------
    Sorted list of key strings relative to the store's prefix.
    """
    pages = store.list(prefix if prefix else None)
    keys = sorted(obj["path"] for page in pages for obj in page)
    log.debug("list_s3 found %d keys under prefix=%r", len(keys), prefix)
    return keys


def open_file(fs, s3_url: str):
    """Open an S3 URL as a file-like object via obstore fsspec.

    Parameters
    ----------
    fs:
        An ``obstore.fsspec.FsspecStore`` returned by :func:`make_fs`.
    s3_url:
        Full S3 URI, e.g. ``s3://bucket/path/to/file.h5``.

    Returns
    -------
    File-like object suitable for ``xr.open_dataset``, ``h5py.File``, etc.
    """
    return fs.open(s3_url)


def handler_from_s3(handler_cls, s3_url: str, fs=None):
    """Instantiate a JoinFileHandler subclass for an S3 object.

    Uses :meth:`JoinFileHandler.from_path` with the obstore fsspec store so
    that ``handler.filename`` is an ``FSFile`` wrapping the obstore backend.

    Parameters
    ----------
    handler_cls:
        A :class:`~join_scratch.datasets.base.JoinFileHandler` subclass.
    s3_url:
        Full S3 URI.
    fs:
        An ``obstore.fsspec.FsspecStore``.  If *None*, one is created with
        :func:`make_fs`.

    Returns
    -------
    Instance of *handler_cls*.
    """
    if fs is None:
        fs = make_fs()
    return handler_cls.from_path(s3_url, fs=fs)
