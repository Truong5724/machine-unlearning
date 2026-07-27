#!/bin/bash
# init.sh - Init SISA UTKFace with random + class unlearning support

set -eou pipefail
IFS=$'\n\t'


if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "$0 <number_of_shards> [unlearn_task] [unlearn_classes]"
    echo ""
    echo "Examples:"
    echo "$0 5"
    echo "$0 5 gender 0"
    echo "$0 5 gender \"0 1\""
    echo "$0 5 race \"1 2\""
    exit 1
fi


shards=$1

# Có thể truyền nhiều điều kiện:
# gender:0 age:2 race:1,3
conditions=("${@:2}")


echo "================================================================="
echo "🚀 Init SISA UTKFace - ${shards} shards"
echo "================================================================="


# Check dataset
[[ -f datasets/UTKFace/datasetfile_ver2 ]] || {
    echo "❌ Datasetfile not found!"
    exit 1
}


mkdir -p containers/utkface/{cache,times,outputs}


echo "📦 Creating shards..."


# ==========================================================
# CREATE SHARDS
# ==========================================================

python distribution_multitask.py \
    --shards "${shards}" \
    --distribution uniform \
    --container utkface \
    --dataset datasets/UTKFace/datasetfile_ver2 \
    --label 0



# ==========================================================
# CREATE REQUEST FILES
# ==========================================================


if [[ ${#conditions[@]} -gt 0 ]]; then

    echo "🎯 Multi-task class unlearning"

    python_args=()

    label="forget"

    for cond in "${conditions[@]}"; do

        task="${cond%%:*}"
        classes="${cond##*:}"

        label="${label}_${task}_${classes//,/_}"

        python_args+=(
            --unlearn
            "${task}:${classes}"
        )

        echo "  ${task} -> ${classes}"

    done

    python distribution_multitask.py \
        --requests 1 \
        --distribution uniform \
        --container utkface \
        --dataset datasets/UTKFace/datasetfile_ver2 \
        --label "${label}" \
        "${python_args[@]}"

    echo "✅ Created class unlearning request"

else

    echo "🎲 Random unlearning mode"


    for req in 0 100 500; do

        python distribution_multitask.py \
            --requests "${req}" \
            --distribution uniform \
            --container utkface \
            --dataset datasets/UTKFace/datasetfile_ver2 \
            --label "${req}"

        echo "✅ Created requestfile for ${req} samples"

    done

fi


echo "================================================================="
echo "✅ Init completed successfully!"
echo "Next step: ./train.sh ${shards}"
echo "================================================================="