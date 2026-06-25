#!/usr/bin/env bash

from obstore.store import S3Store
import icechunk as ic
import boto3

import os
from functools import partial


def ic_s3_static_creds(region: str = "us-west-2") -> ic.credentials.S3StaticCredentials:
    """
    Create an icechunk S3StaticCredentials object from AWS environment 
    credentials retrieved by boto3.

    Args:
        region (str): AWS region for the credentials (default = "us-west-2")
    """
    session = boto3.Session(region_name=region)
    creds = session.get_credentials()
    if creds is None:
        raise ValueError("Could not find valid AWS credentials")

    frozen = creds.get_frozen_credentials()
    return ic.credentials.S3StaticCredentials(
        access_key_id=frozen.access_key,
        secret_access_key=frozen.secret_key,
        session_token=frozen.token,
    )


def ic_s3_credentials(**kwargs):
    """
    Create an icechunk s3_credentials object from AWS environment credentials 
    retrieved via boto3. Intended as a convenience wrapper for use in 
    `icechunk.credentials.containers_credentials`.
    """
    return ic.credentials.s3_credentials(
        get_credentials=partial(ic_s3_static_creds, **kwargs)
    )


def s3_store_config(region: str = "us-west-2") -> dict: 
    """Return an obstore for use with xarray / h5py / satpy."""

    # 1. Hide empty environment variables from obstore's Rust backend
    for bad_var in ["AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN"]:
        if os.environ.get(bad_var) == "":
            del os.environ[bad_var]

    # 2. Use boto3 to grab your working credentials (handles profiles, SSO, ~/.aws, etc.)
    session = boto3.Session(region_name=region)
    creds = session.get_credentials()

    config = {"region": region}

    # 3. If boto3 found credentials, extract the raw strings
    if creds:
        frozen = creds.get_frozen_credentials()
        config["access_key_id"] = frozen.access_key
        config["secret_access_key"] = frozen.secret_key
        if frozen.token:
            config["session_token"] = frozen.token

    # 4. Filter out any empty values so obstore doesn't crash on 'None'
    clean_config = {k: v for k, v in config.items() if v}

    return clean_config


def list_s3(store: S3Store, prefix: str = "") -> list[str]:
    """List all object keys under *prefix* in *store*."""
    pages = store.list(prefix if prefix else None)
    keys = sorted(obj["path"] for page in pages for obj in page)
    return keys

