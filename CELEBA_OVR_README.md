# CelebA OVR (One-vs-Rest) Multi-Attribute Classification System

Hệ thống này chuẩn bị và huấn luyện 27 mô hình nhị phân (OVR) cho CelebA dataset, mỗi mô hình dự báo một thuộc tính cụ thể.

## 📋 Danh Sách 27 Attributes Được Chọn

1. **male** - Giới tính nam
2. **young** - Trẻ tuổi
3. **smiling** - Cười
4. **mouth_slightly_open** - Miệng hơi mở
5. **big_lips** - Môi dày
6. **big_nose** - Mũi to
7. **pointy_nose** - Mũi nhọn
8. **high_cheekbones** - Gò má cao
9. **oval_face** - Gương mặt hình bầu dục
10. **wavy_hair** - Tóc ngoạn cuộn
11. **straight_hair** - Tóc thẳng
12. **bangs** - Tóc mái
13. **receding_hairline** - Chân tóc lùi
14. **black_hair** - Tóc đen
15. **blond_hair** - Tóc vàng
16. **brown_hair** - Tóc nâu
17. **eyeglasses** - Kính mắt
18. **bushy_eyebrows** - Chân mày dày
19. **arched_eyebrows** - Chân mày cung
20. **bags_under_eyes** - Quầng thâm dưới mắt
21. **chubby** - Mặt tròn
22. **double_chin** - Cằm kép
23. **wearing_earrings** - Đeo bông tai
24. **wearing_necklace** - Đeo vòng cổ
25. **mustache** - Ria mép
26. **goatee** - Mép dê
27. **sideburns** - Mái tóc bên má

## 🚀 Quy Trình Sử Dụng

### Bước 1: Chuẩn Bị Dữ Liệu (Stratified Multi-Label Sampling)

Giảm dataset từ 202k → 50k ảnh đã cân bằng:

```bash
cd /home/tri/machine-unlearning

python datasets/celebA/prepare_data_ovr.py \
    --input_dir datasets/celebA/img_align_celeba \
    --attr_file datasets/celebA/list_attr_celeba.txt \
    --output_dir datasets/celebA \
    --train_samples 50000 \
    --test_samples 10000 \
    --seed 42
```

**Tạo ra:**
- `datasets/celebA/celeba_ovr_train.h5` - 50,000 ảnh training (64×64)
- `datasets/celebA/celeba_ovr_test.h5` - 10,000 ảnh test (64×64)
- `datasets/celebA/datasetfile_ovr` - Metadata file

**Lưu ý:** Sử dụng stratified multi-label sampling để đảm bảo:
- Các label combinations xuất hiện cân bằng trong training set
- Dữ liệu giảm xuống ~50k theo tỷ lệ stratified
- Giữ nguyên phân phối label distribution

### Bước 2: Khởi Tạo Partitions Cho SISA Training

Tạo 27 shard (mỗi shard = 1 attribute, 1 OVR model):

```bash
python celeba_ovr_partition.py \
    --container celeba_ovr \
    --dataset datasets/celebA/datasetfile_ovr \
    --label 0 \
    --slices_per_shard 2 \
    --seed 42
```

**Tạo ra:**
- `containers/celeba_ovr/splitfile.npy` - Danh sách indices cho 27 shard
- `containers/celeba_ovr/requestfile:0.npy` - Request file (ban đầu rỗng, cho baseline)
- `containers/celeba_ovr/ovr_slices.npz` - SISA slices (2 slices/shard mặc định)
- `containers/celeba_ovr/ovr_meta.json` - Metadata (task mapping)
- `containers/celeba_ovr/cache/` - Thư mục lưu checkpoint
- `containers/celeba_ovr/times/` - Thư mục lưu thời gian training
- `containers/celeba_ovr/outputs/` - Thư mục lưu kết quả

### Bước 3: Huấn Luyện Các Models OVR

Huấn luyện từng shard (attribute) theo SISA framework:

```bash
# Shard 0: male
python sisa_celeba_ovr.py \
    --container celeba_ovr \
    --shard 0 \
    --dataset datasets/celebA/datasetfile_ovr \
    --label 0 \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --optimizer adam \
    --dropout_rate 0.3 \
    --loss_mode auto

# Shard 1: young
python sisa_celeba_ovr.py \
    --container celeba_ovr \
    --shard 1 \
    --dataset datasets/celebA/datasetfile_ovr \
    --label 0 \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 0.001
```

**Tham số quan trọng:**
- `--shard`: ID shard (0..26)
- `--epochs`: Số epoch cho mỗi slice
- `--batch_size`: Batch size (32 khuyến cáo)
- `--learning_rate`: LR (0.001 mặc định)
- `--loss_mode`: 'auto' = dùng focal loss cho imbalanced tasks
- `--optimizer`: 'adam' hoặc 'sgd'
- `--dropout_rate`: Dropout (0.3 mặc định)

**Để chạy song song tất cả 27 shards:**

```bash
for shard in {0..26}; do
    python sisa_celeba_ovr.py \
        --container celeba_ovr \
        --shard $shard \
        --dataset datasets/celebA/datasetfile_ovr \
        --label 0 \
        --epochs 5 \
        --batch_size 32 \
        --learning_rate 0.001 &
done
wait
```

### Bước 4: Chuẩn Bị Unlearning Requests

Tạo request file để unlearn một slice dữ liệu:

```bash
# Unlearn slice 0 của attribute "smiling"
python celeba_ovr_make_requestfile.py \
    --container celeba_ovr \
    --label forget-smiling-slice0 \
    --task smiling \
    --slice 0 \
    --mode overwrite
```

**Tham số:**
- `--task`: Attributes trong danh sách OVR_TASKS
- `--slice`: ID slice (0 hoặc 1 nếu slices_per_shard=2)
- `--mode`: 'overwrite' = tạo mới, 'merge' = union với request cũ

**Để merge nhiều requests:**

```bash
# Request 1
python celeba_ovr_make_requestfile.py \
    --container celeba_ovr \
    --label combined-request \
    --task smiling \
    --slice 0 \
    --mode overwrite

# Request 2 (merge)
python celeba_ovr_make_requestfile.py \
    --container celeba_ovr \
    --label combined-request \
    --task young \
    --slice 1 \
    --mode merge
```

### Bước 5: Huấn Luyện Với Unlearning

Huấn luyện lại các shards theo request:

```bash
# Huấn luyện lại shard=3 (attribute=smiling) với unlearning
python sisa_celeba_ovr.py \
    --container celeba_ovr \
    --shard 3 \
    --dataset datasets/celebA/datasetfile_ovr \
    --label forget-smiling-slice0 \
    --epochs 5 \
    --batch_size 32
```

## 📂 Cấu Trúc File Được Tạo

```
datasets/celebA/
├── prepare_data_ovr.py          # Script chuẩn bị dữ liệu
├── dataloader_ovr.py             # Dataloader OVR
├── celeba_ovr_train.h5          # Training set HDF5 (50k ảnh)
├── celeba_ovr_test.h5           # Test set HDF5 (10k ảnh)
└── datasetfile_ovr              # Metadata JSON

architectures/
├── celeba_ovr.py                # Model architecture & OVR_TASKS

containers/celeba_ovr/
├── splitfile.npy                # Indices cho 27 shards
├── requestfile:0.npy            # Request file cho baseline
├── requestfile:*:label.npy      # Request files khác
├── ovr_slices.npz               # SISA slices
├── ovr_meta.json                # Task metadata
├── cache/                        # Checkpoints
├── times/                        # Training times
└── outputs/                      # Results & predictions

./
├── celeba_ovr_partition.py       # Khởi tạo partition
├── sisa_celeba_ovr.py            # Training script
└── celeba_ovr_make_requestfile.py # Unlearning request script
```

## 🔧 Chi Tiết Kỹ Thuật

### Stratified Multi-Label Sampling

`prepare_data_ovr.py` sử dụng stratified sampling để:
1. Tạo binary string representation cho mỗi sample (vd: "10101...")
2. Tìm unique label combinations
3. Sample từ mỗi stratum theo tỷ lệ (~target_samples / total_samples)

**Lợi ích:**
- Cân bằng dữ liệu: mỗi label combination xuất hiện tương đương
- Giảm kích thước dataset từ 202k → 50k
- Giữ nguyên distribution các label hiếm

### SISA Training

- **Shards**: 27 shard, mỗi shard = 1 attribute OVR
- **Slices**: Mỗi shard chia thành 2 slice (mặc định)
- **Sequential Training**: Train slice 0 → slice 0+1 → slice 0+1+2...
- **Checkpoint**: Lưu intermediate models giữa slices

### Loss Functions

- **Binary Cross Entropy (BCE)**: Mặc định cho balanced data
- **Focal Loss**: Tự động cho imbalanced tasks (pos_weight < 0.5)
  - `gamma=2.0`: Focusing parameter
  - `alpha`: Automatic dựa trên class imbalance ratio

## 📊 Monitoring

Kiểm tra tiến trình training:

```bash
# Xem logs
ls -lh containers/celeba_ovr/cache/

# Kiểm tra checkpoint cho shard cụ thể
ls containers/celeba_ovr/cache/shard-3:0.pt

# Xem metadata
cat containers/celeba_ovr/ovr_meta.json

# Xem request details
cat containers/celeba_ovr/requestfile:forget-smiling-slice0.json
```

## ⚠️ Lưu Ý quan Trọng

1. **GPU Memory**: Batch size 32 cần ~4-6GB VRAM, điều chỉnh theo GPU của bạn
2. **Training Time**: 27 shards × 5 epochs ≈ 2-4 giờ trên GPU
3. **Seed Reproducibility**: Luôn dùng `--seed 42` để reproducible
4. **Request Files**: Không được xóa, cần giữ lại cho tracking unlearning
5. **Checkpoint Symlinks**: Không xóa symlinks (`shard-*:*.pt`), chúng trỏ tới actual checkpoint

## 🐛 Troubleshooting

**Lỗi: "Không tìm thấy celeba_ovr_train.h5"**
→ Chạy `prepare_data_ovr.py` trước

**Lỗi: "Missing ovr_slices.npz"**
→ Chạy `celeba_ovr_partition.py` trước

**Lỗi: CUDA Out of Memory**
→ Giảm `--batch_size` hoặc `--epochs`

**Lỗi: "Unsupported OVR task"**
→ Kiểm tra tên task trong OVR_TASKS (tất cả chữ thường)

## 📝 Example Workflow

```bash
# 1. Chuẩn bị dữ liệu (lần đầu, mất ~30 phút)
python datasets/celebA/prepare_data_ovr.py \
    --train_samples 50000 --seed 42

# 2. Khởi tạo partitions
python celeba_ovr_partition.py \
    --container celeba_ovr --label 0 --seed 42

# 3. Huấn luyện tất cả 27 shards (song song)
for shard in {0..26}; do
    python sisa_celeba_ovr.py \
        --container celeba_ovr \
        --shard $shard \
        --dataset datasets/celebA/datasetfile_ovr \
        --label 0 \
        --epochs 5 &
done
wait

# 4. Tạo unlearning request
python celeba_ovr_make_requestfile.py \
    --container celeba_ovr \
    --label forget-young \
    --task young \
    --slice 0

# 5. Huấn luyện lại shard tương ứng với unlearning
python sisa_celeba_ovr.py \
    --container celeba_ovr \
    --shard 1 \
    --label forget-young \
    --epochs 5
```

## 📚 References

- **Paper**: SISA Training (Sharded, Isolated, Sliced, Aggregated)
- **Related**: UTKFace OVR implementation (`sisa_utkface_ovr.py`)
- **Stratified Sampling**: scikit-learn pipeline

---
**Tác giả**: Được tạo tự động cho CelebA OVR multi-attribute classification
**Cập nhật**: 2026-03-25
