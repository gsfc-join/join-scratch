#!/usr/bin/env bash

from obstore.store import S3Store
import boto3
import os

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

