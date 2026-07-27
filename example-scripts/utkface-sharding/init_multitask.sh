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

# Optional class unlearning parameters
unlearn_task=${2:-}
unlearn_classes=${3:-}


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


if [[ -n "${unlearn_task}" && -n "${unlearn_classes}" ]]; then

    echo "🎯 Class unlearning mode"
    echo "Task   : ${unlearn_task}"
    echo "Class  : ${unlearn_classes}"


    # convert "0 1" -> arguments 0 1
    class_args=(${unlearn_classes})


    python distribution_multitask.py \
        --requests 1 \
        --distribution uniform \
        --container utkface \
        --dataset datasets/UTKFace/datasetfile_ver2 \
        --label forget_${unlearn_task}_${unlearn_classes// /_} \
        --unlearn_task "${unlearn_task}" \
        --unlearn_class "${class_args[@]}"


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