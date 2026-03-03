#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

container=${1:-utkface}
label=${2:-forget-gender}
shard=${3:-0}

# Backward-compatible: unlearn a WHOLE slice (recommended)
# Usage examples:
#   bash unlearn_shard_multitask.sh utkface forget-gender-slice1 0 1
#   bash unlearn_shard_multitask.sh utkface forget-age-slice2    1 2
#   bash unlearn_shard_multitask.sh utkface forget-race-slice4   2 4
slice=${4:-0}

python utkface_multitask_make_requestfile.py \
  --container ${container} \
  --label ${label} \
  --shard ${shard} \
  --slice ${slice} \
  --mode overwrite

echo "✅ Unlearn request applied: shard=${shard}, slice=${slice}, label=${label}"
