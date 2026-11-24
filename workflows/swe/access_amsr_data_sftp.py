#!/usr/bin/env python3
"""
JAXA G-Portal L1B Downloader
Downloads AMSR-2 (1BGCOMW1AMSR2) and/or AMSR-3 (1BGOSATGWAMSR3) Level 1B files
over SFTP for a specified date range.
"""

import os
import logging
from datetime import timedelta

import fsspec
import pandas as pd

log = logging.getLogger("gportal_l1b_downloader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Confirmed AMSR2 path (no day-level subdir — files for the whole month live
# in {year}/{month}).
# AMSR3 path is a PLACEHOLDER — confirm/update once published on G-Portal SFTP.
REMOTE_PATH_TEMPLATES = {
    "1BGCOMW1AMSR2": "/product/Standard/GCOM-W/GCOM-W.AMSR2/L1B/2/{year}/{month}",
    "1BGOSATGWAMSR3": "/product/Standard/GOSAT-GW/GOSAT-GW.AMSR3/L1B/2/{year}/{month}",  # UNCONFIRMED
}

# Glob patterns derived from confirmed filename examples:
#   GW1AM2_201501312324_168A_L1SGRTBR_2220220.h5
#   GGWAM3_202609062255D017_S1BTBBGAZ01A26250.nc
# The wildcard after the 8-digit date absorbs the HHMM timestamp + orbit/pass ID.
FILENAME_PATTERNS = {
    "1BGCOMW1AMSR2": "GW1AM2_{date_compact}*_L1SGBTBR_*.h5",
    "1BGOSATGWAMSR3": "GGWAM3_{date_compact}*_S1BTBBGAZ*.nc",
}

PRODUCT_AVAILABLE = {
    "1BGCOMW1AMSR2": True,
    "1BGOSATGWAMSR3": False,  # Not yet available on SFTP as of 2026-11-20
}


def get_sftp_connection():
    """Authenticate to JAXA G-Portal SFTP using a private key (interactive prompts)."""
    print("=== JAXA G-Portal Authentication ===")
    sftp_host = "sftp.gportal.jaxa.jp"

    sftp_user = input("Enter JAXA G-Portal Username: ").strip()
    if not sftp_user:
        print("Error: Username cannot be empty.")
        return None

    private_key = input("Private Key Path [/home/jovyan/privatekey.key]: ").strip() \
        or "/home/jovyan/privatekey.key"

    if not os.path.exists(private_key):
        print(f"Error: Private key file not found at {private_key}")
        return None

    print("Connecting to SFTP using private key...")
    try:
        sftp_fs = fsspec.filesystem(
            "sftp",
            host=sftp_host,
            username=sftp_user,
            key_filename=private_key,
            look_for_keys=False,  # Strictly use the provided key
        )
        sftp_fs.ls("/")
        print("SFTP Connection Successful.")
        return sftp_fs
    except Exception as e:
        print(f"Failed to connect to SFTP: {e}")
        return None


def find_remote_files(fs, product_code: str, current_date: pd.Timestamp) -> list[str]:
    """Build the remote dir + glob for a given product/date and return matches."""
    if product_code not in REMOTE_PATH_TEMPLATES:
        log.error(f"Unknown product code: {product_code}")
        return []

    if not PRODUCT_AVAILABLE.get(product_code, True):
        log.warning(f"[{product_code}] Product not yet available on G-Portal SFTP. Skipping.")
        return []

    year_str = current_date.strftime("%Y")
    month_str = current_date.strftime("%m")
    date_compact = current_date.strftime("%Y%m%d")  # 8-digit day prefix

    remote_dir = REMOTE_PATH_TEMPLATES[product_code].format(year=year_str, month=month_str)
    pattern = FILENAME_PATTERNS[product_code].format(date_compact=date_compact)
    search_pattern = f"{remote_dir.rstrip('/')}/{pattern}"

    log.info(f"[{product_code}] Searching: {search_pattern}")
    try:
        matches = fs.glob(search_pattern)
        if not matches:
            log.warning(f"[{product_code}] No files found for {date_compact} in {remote_dir}")
        return matches
    except FileNotFoundError:
        log.warning(f"[{product_code}] Remote directory does not exist: {remote_dir}")
        return []
    except Exception as e:
        log.error(f"[{product_code}] Error during glob search: {e}")
        return []


def download_file(fs, remote_path: str, local_dir: str, overwrite: bool = False) -> bool:
    """Download a single remote file to local_dir, preserving the filename."""
    filename = os.path.basename(remote_path)
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path) and not overwrite:
        log.info(f"Already exists, skipping: {local_path}")
        return True

    os.makedirs(local_dir, exist_ok=True)
    try:
        log.info(f"Downloading {remote_path} -> {local_path}")
        fs.get(remote_path, local_path)
        return True
    except Exception as e:
        log.error(f"Failed to download {remote_path}: {e}")
        return False


def download_l1b_range(
    fs,
    product_codes: list[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    overwrite: bool = False,
) -> dict[str, list[str]]:
    """
    Download L1B files for one or more product codes over a date range.

    Since AMSR2's remote directory is organized by {year}/{month} only (all
    days together), we glob once per day using a date-stamped filename
    pattern rather than navigating into day subfolders.

    Returns a dict mapping product_code -> list of local file paths downloaded.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    results: dict[str, list[str]] = {p: [] for p in product_codes}

    current_date = start
    while current_date <= end:
        for product_code in product_codes:
            remote_files = find_remote_files(fs, product_code, current_date)
            product_out_dir = os.path.join(output_dir, product_code, current_date.strftime("%Y%m%d"))

            for remote_file in remote_files:
                success = download_file(fs, remote_file, product_out_dir, overwrite=overwrite)
                if success:
                    results[product_code].append(
                        os.path.join(product_out_dir, os.path.basename(remote_file))
                    )

        current_date += timedelta(days=1)

    return results


def main():
    fs = get_sftp_connection()
    if fs is None:
        return

    print("\n=== L1B Download Parameters ===")
    start_date = input("Start date (YYYY-MM-DD): ").strip()
    end_date = input("End date (YYYY-MM-DD): ").strip()
    output_dir = input("Local output directory [./l1b_downloads]: ").strip() or "./l1b_downloads"

    print("\nSelect product(s) to download:")
    print("  1) AMSR-2 (1BGCOMW1AMSR2)")
    print("  2) AMSR-3 (1BGOSATGWAMSR3) — NOTE: not yet available on SFTP")
    print("  3) Both")
    choice = input("Choice [1]: ").strip() or "1"

    if choice == "1":
        product_codes = ["1BGCOMW1AMSR2"]
    elif choice == "2":
        product_codes = ["1BGOSATGWAMSR3"]
    else:
        product_codes = ["1BGCOMW1AMSR2", "1BGOSATGWAMSR3"]

    overwrite_input = input("Overwrite existing local files? [y/N]: ").strip().lower()
    overwrite = overwrite_input == "y"

    results = download_l1b_range(
        fs=fs,
        product_codes=product_codes,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    print("\n=== Download Summary ===")
    for product_code, files in results.items():
        status = "" if PRODUCT_AVAILABLE.get(product_code, True) else "  (product not yet available)"
        print(f"{product_code}: {len(files)} file(s) downloaded to {os.path.join(output_dir, product_code)}{status}")


if __name__ == "__main__":
    main()