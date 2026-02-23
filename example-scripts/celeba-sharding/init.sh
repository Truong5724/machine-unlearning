#!/bin/bash
# init_optimized.sh - Khởi tạo SISA container cho CelebA với HDF5

set -eou pipefail
IFS=$'\n\t'

shards=$1

echo "======================================================================"
echo "KHỞI TẠO SISA CONTAINER CHO CELEBA"
echo "======================================================================"
echo "Số shards: ${shards}"
echo ""

# Kiểm tra datasetfile tồn tại
if [[ ! -f "datasets/celebA/datasetfile" ]]; then
    echo "❌ KHÔNG TÌM THẤY datasets/celebA/datasetfile"
    echo "   Hãy chạy prepare_data.py trước!"
    echo ""
    echo "   cd datasets/celebA"
    echo "   python prepare_data.py --attribute Smiling --batch_size 1000"
    exit 1
fi

# Kiểm tra HDF5 files
if [[ ! -f "datasets/celebA/celeba_train.h5" ]]; then
    echo "❌ KHÔNG TÌM THẤY datasets/celebA/celeba_train.h5"
    echo "   Hãy chạy prepare_data.py trước!"
    exit 1
fi

echo "✅ Dataset files OK"
echo ""

# Tạo container structure
if [[ ! -d "containers/celeba" ]] ; then
    echo "📁 Tạo thư mục container..."
    mkdir -p "containers/celeba"
    mkdir -p "containers/celeba/cache"
    mkdir -p "containers/celeba/times"
    mkdir -p "containers/celeba/outputs"
    mkdir -p "containers/celeba/shards"
    echo 0 > "containers/celeba/times/null.time"
    echo "✅ Đã tạo container structure"
else
    echo "✅ Container đã tồn tại"
fi

echo ""
echo "🔄 Chia data thành ${shards} shards..."
python distribution.py --shards "${shards}" --distribution uniform \
    --container "celeba" \
    --dataset datasets/celebA/datasetfile \
    --label 0

echo "✅ Đã tạo ${shards} shards"
echo ""

# # Tạo unlearning request scenarios
# echo "🔄 Tạo 15 unlearning request scenarios..."
# for j in {1..15}; do
#     r=$((${j}*${shards}/5))
#     echo "  Scenario ${j}/15: ${r} requests"
#     python distribution.py --requests "${r}" --distribution uniform \
#         --container "celeba" \
#         --dataset datasets/celebA/datasetfile \
#         --label "${r}"
# done