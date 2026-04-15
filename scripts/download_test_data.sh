#!/usr/bin/env bash
# Download the LIS inputs file and 2 sample files from each data subdirectory
# into _data-raw/, preserving the S3 key structure.

set -euo pipefail

BUCKET="s3://airborne-smce-prod-user-bucket/JOIN"
DEST="_data-raw"

# ---------------------------------------------------------------------------
# LIS inputs
# ---------------------------------------------------------------------------
if [[ ! -f "${DEST}/lis_input_NMP_1000m_missouri.nc" ]]; then
  aws s3 cp \
    "${BUCKET}/lis_input_NMP_1000m_missouri.nc" \
    "${DEST}/lis_input_NMP_1000m_missouri.nc"
fi

# ---------------------------------------------------------------------------
# Helper: download the first N files found under a given S3 prefix
# ---------------------------------------------------------------------------
download_n() {
  local prefix="$1"
  local n="$2"

  aws s3 ls --recursive "${prefix}" \
    | awk '{print $4}' \
    | head -"${n}" \
    | while IFS= read -r key; do
        DESTFILE="${DEST}/${key}"
        [[ -f $DESTFILE ]] && echo "Skipping existing file $DESTFILE" && continue
        aws s3 cp \
          "s3://airborne-smce-prod-user-bucket/${key}" \
          "${DEST}/${key}"
      done
}

download_n "${BUCKET}/AMSR2/" 2
download_n "${BUCKET}/CEDA/" 2
download_n "${BUCKET}/ICESAT-2/" 2
download_n "${BUCKET}/VIIRS/" 2

echo "Done."
