# #!/bin/bash
# # init_fast.sh - Khởi tạo SISA với ÍT SCENARIOS (cho Colab Free)

# set -eou pipefail
# IFS=$'\n\t'

# shards=$1

# echo "======================================================================"
# echo "KHỞI TẠO SISA - FAST MODE (ÍT SCENARIOS)"
# echo "======================================================================"
# echo "Shards: ${shards}"
# echo "Scenarios: 4 (thay vì 16) → Tiết kiệm 75% thời gian!"
# echo ""

# # Kiểm tra dataset
# if [[ ! -f "datasets/UTKFace/datasetfile" ]]; then
#     echo "❌ KHÔNG TÌM THẤY datasets/UTKFace/datasetfile"
#     exit 1
# fi

# if [[ ! -f "datasets/UTKFace/utkface_train.h5" ]]; then
#     echo "❌ KHÔNG TÌM THẤY datasets/UTKFace/utkface_train.h5"
#     exit 1
# fi

# echo "✅ Dataset OK"
# echo ""

# # Tạo container
# if [[ ! -d "containers/utkface" ]] ; then
#     echo "📁 Tạo thư mục container..."
#     mkdir -p "containers/utkface"
#     mkdir -p "containers/utkface/cache"
#     mkdir -p "containers/utkface/times"
#     mkdir -p "containers/utkface/outputs"
#     mkdir -p "containers/utkface/shards"
#     echo 0 > "containers/utkface/times/null.time"
# fi

# echo "🔄 Chia data thành ${shards} shards..."
# python distribution.py --shards "${shards}" --distribution uniform \
#     --container "utkface" \
#     --dataset datasets/UTKFace/datasetfile \
#     --label 0

# echo "✅ Đã tạo ${shards} shards"
# echo ""

# # Tạo SELECTIVE unlearning scenarios (chỉ 4 scenarios)
# echo "🔄 Tạo 4 unlearning scenarios (thay vì 16)..."

# # Chọn scenarios: 0, 5, 10, 15
# scenarios=(0 5 10 15)

# for j in "${scenarios[@]}"; do
#     if [ $j -eq 0 ]; then
#         echo "  Scenario 1/4: baseline (0 requests)"
#         continue
#     fi
    
#     r=$((${j}*${shards}/5))
#     echo "  Scenario $((${j}/5 + 1))/4: ${r} requests"
#     python distribution.py --requests "${r}" --distribution uniform \
#         --container "utkface" \
#         --dataset datasets/UTKFace/datasetfile \
#         --label "${r}"
# done

# echo ""
# echo "======================================================================"
# echo "✅ KHỞI TẠO HOÀN TẤT - FAST MODE!"
# echo "======================================================================"
# echo "Shards: ${shards}"
# echo "Scenarios: 4 (0%, 33%, 67%, 100% unlearn)"
# echo ""
# echo "⏱️  Thời gian ước tính training:"
# echo "   ${shards} shards × 4 scenarios × 1 giờ = $((${shards} * 4)) giờ"
# echo ""
# echo "Bước tiếp theo:"
# echo "  ./train_fast.sh ${shards}"
# echo "======================================================================"
!/bin/bash
init_utkface_simple.sh - Khởi tạo SISA CHỈ TRAIN (KHÔNG UNLEARNING)

set -eou pipefail
IFS=$'\n\t'

shards=$1

echo "======================================================================"
echo "KHỞI TẠO UTKFace - SIMPLE MODE (KHÔNG UNLEARNING)"
echo "======================================================================"
echo "Shards: ${shards}"
echo ""

# ===============================
# 1️⃣ Kiểm tra dataset
# ===============================
if [[ ! -f "datasets/UTKFace/datasetfile" ]]; then
    echo "❌ Không tìm thấy datasets/UTKFace/datasetfile"
    exit 1
fi

if [[ ! -f "datasets/UTKFace/utkface_train.h5" ]]; then
    echo "❌ Không tìm thấy datasets/UTKFace/utkface_train.h5"
    exit 1
fi

echo "✅ Dataset OK"
echo ""

# ===============================
# 2️⃣ Tạo container nếu chưa có
# ===============================
if [[ ! -d "containers/utkface" ]]; then
    echo "📁 Tạo container..."
    mkdir -p containers/utkface/{cache,times,outputs,shards}
    echo 0 > containers/utkface/times/null.time
fi

# ===============================
# 3️⃣ Chia shard (KHÔNG tạo request)
# ===============================
echo "🔄 Chia data thành ${shards} shards..."

python distribution.py \
    --shards "${shards}" \
    --distribution uniform \
    --container "utkface" \
    --dataset datasets/UTKFace/datasetfile \
    --label 0

echo ""
echo "======================================================================"
echo "✅ KHỞI TẠO HOÀN TẤT - SIMPLE MODE"
echo "======================================================================"
echo "Shards: ${shards}"
echo ""
echo "Bước tiếp theo:"
echo "  ./train_utkface.sh ${shards}"
echo "======================================================================"