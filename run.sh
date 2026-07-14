#!/usr/bin/env bash
set -ex

# Beyond Spatial Resolution — Code Ocean entry point
#
# The working directory during execution is /code (where this script lives).
# Input data  : /data   (Datasets/ CSVs and optionally Imagery/ GeoTIFFs)
# Output data : /results
#
# Any extra arguments are forwarded to run.py (e.g. --no-image, --models RF).

python run.py "$@"
